#!/usr/bin/env python
"""
Aperture photometry of extended objects (galaxies) in an r-band (Sloan) image.

Pipeline
--------
1. Estimate and subtract a 2D background (photutils.background.Background2D).
2. Detect sources on the smoothed image (segmentation + deblending).
3. Separate point sources (stars) from extended objects using the source size
   relative to the PSF FWHM measured from the field itself.
4. Mask stars (bright ones get a larger mask) so they do not contaminate the
   galaxy apertures.
5. Elliptical aperture photometry of every extended object, with a local
   sky annulus and a correction for the masked fraction of the aperture.
6. Match a few stars to a reference list (RA/Dec or x/y plus r magnitude)
   and compute the photometric zero point  ZP = m_ref + 2.5 log10(F_star / t).
7. Convert galaxy fluxes into apparent r magnitudes:  m = ZP - 2.5 log10(F / t).

Usage
-----
    python galaxy_photometry.py image.fits --refstars refs.csv --exptime 120 \
        --out galaxies.ecsv --plot field.png

    refs.csv (header required), either sky or pixel coordinates:
        ra,dec,mag_r          (degrees; needs a WCS in the FITS header)
        x,y,mag_r             (pixels, 0-indexed)

    python galaxy_photometry.py --demo        # run on a synthetic image

Image units are assumed to be counts (ADU or e-) unless --exptime 1 is given
for an image already in counts/s. The result is the Sloan r (AB-like) apparent
magnitude on the same system as the reference magnitudes you provide.
"""
import argparse

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                ApertureStats, aperture_photometry)
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import (SourceCatalog, deblend_sources,
                                    detect_sources, detect_threshold)
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve


# --------------------------------------------------------------------------
# 1-3. Background, detection, star/galaxy separation
# --------------------------------------------------------------------------
def subtract_background(data, box_size=64, filter_size=3, source_mask=None):
    """Return background-subtracted image and the Background2D object."""
    bkg = Background2D(data, box_size, filter_size=filter_size,
                       mask=source_mask, sigma_clip=SigmaClip(sigma=3.0),
                       bkg_estimator=MedianBackground())
    return data - bkg.background, bkg


def detect_objects(data, nsigma=3.0, npixels=15, kernel_fwhm=3.0,
                   deblend=True, contrast=0.005):
    """Segmentation-based detection on a smoothed image.

    Returns the segmentation map, the smoothed image and a SourceCatalog.
    A first pass is used to mask sources and get a cleaner background.
    """
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=int(4 * kernel_fwhm) | 1)
    conv = convolve(data, kernel, normalize_kernel=True)
    threshold = detect_threshold(data, n_sigma=nsigma)
    segm = detect_sources(conv, threshold, n_pixels=npixels)
    if segm is None:
        raise RuntimeError("No sources detected; lower --nsigma/--npixels.")
    if deblend:
        segm = deblend_sources(conv, segm, n_pixels=npixels, n_levels=32,
                               contrast=contrast, progress_bar=False)
    cat = SourceCatalog(data, segm, convolved_data=conv)
    return segm, conv, cat


def classify_sources(cat, ext_factor=1.5, min_area=40):
    """Split catalog into stars / extended objects using size vs PSF FWHM.

    The PSF FWHM is estimated from the compact end of the bright sources.
    Returns (is_extended, psf_fwhm) where is_extended is a boolean array.
    """
    fwhm = np.asarray(cat.fwhm.value, float)
    flux = np.asarray(cat.segment_flux, float)
    area = np.asarray(cat.area.value, float)
    good = np.isfinite(fwhm) & (flux > 0)
    # PSF: smallest 30% (by fwhm) among the brighter half of the sources
    bright = good & (flux >= np.nanmedian(flux[good]))
    cand = np.sort(fwhm[bright])
    psf_fwhm = np.median(cand[:max(3, int(0.3 * cand.size))])
    is_ext = good & (fwhm > ext_factor * psf_fwhm) & (area >= min_area)
    return is_ext, float(psf_fwhm)


# --------------------------------------------------------------------------
# 4. Star masking
# --------------------------------------------------------------------------
def build_star_mask(shape, cat, star_idx, psf_fwhm, bright_frac=0.2,
                    small_radius=2.5, big_radius=5.0, saturation=None):
    """Boolean mask (True = masked) with a disk on every star.

    The brightest `bright_frac` of the stars (or saturated ones) get a larger
    disk, to cover their wings and diffraction spikes.
    """
    mask = np.zeros(shape, bool)
    if len(star_idx) == 0:
        return mask
    flux = np.asarray(cat.segment_flux)[star_idx]
    peak = np.asarray(cat.max_value)[star_idx]
    big = flux >= np.quantile(flux, 1 - bright_frac)
    if saturation is not None:
        big |= peak >= saturation
    yy, xx = np.indices(shape)
    xc = np.asarray(cat.x_centroid)[star_idx]
    yc = np.asarray(cat.y_centroid)[star_idx]
    for x, y, b in zip(xc, yc, big):
        r = (big_radius if b else small_radius) * psf_fwhm
        mask |= (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    return mask


# --------------------------------------------------------------------------
# 5. Aperture photometry
# --------------------------------------------------------------------------
def _masked_area_correction(aperture, mask):
    """Total aperture area / unmasked aperture area (>= 1)."""
    ap_mask = aperture.to_mask(method='center')
    w = ap_mask.get_values(np.ones_like(mask, float))  # weights inside box
    m = ap_mask.get_values(mask.astype(float))
    total = w.size
    free = np.sum(m == 0)
    return total / free if free > 0 else np.nan


def photometer_galaxies(data, cat, ext_idx, mask, aper_scale=4.0,
                        annulus_gap=1.5, annulus_width=1.5, sky_mask=None):
    """Elliptical-aperture photometry of the extended objects.

    Aperture axes are `aper_scale` times the second-moment semi-axes (the
    ellipse enclosing most of the light; adjust to taste), with a local sky annulus. Pixels in
    `mask` (stars, other segments) are excluded and the flux is rescaled for
    the lost area. Returns an astropy Table (flux in counts, not per second).
    """
    rows = []
    sky_mask = mask if sky_mask is None else sky_mask
    for i in ext_idx:
        x, y = cat.x_centroid[i], cat.y_centroid[i]
        a = aper_scale * cat.semimajor_axis.value[i]
        b = aper_scale * cat.semiminor_axis.value[i]
        theta = cat.orientation.to('rad').value[i]
        if not np.all(np.isfinite([x, y, a, b, theta])) or b <= 0:
            continue
        ap = EllipticalAperture((x, y), a, b, theta=theta)
        ann = EllipticalAnnulus((x, y), a * annulus_gap,
                                a * (annulus_gap + annulus_width),
                                b * (annulus_gap + annulus_width), b_in=b * annulus_gap,
                                theta=theta)
        sky_stats = ApertureStats(data, ann, mask=sky_mask,
                                  sigma_clip=SigmaClip(3.0))
        sky = sky_stats.median if np.isfinite(sky_stats.median) else 0.0
        tab = aperture_photometry(data - sky, ap, mask=mask)
        raw = float(tab['aperture_sum'][0])
        corr = _masked_area_correction(ap, mask)
        flux = raw * corr
        npix = ap.area * 1.0
        # Poisson-free background-dominated uncertainty (sky rms * sqrt(Npix))
        err = sky_stats.std * np.sqrt(npix) * np.sqrt(corr)
        rows.append(dict(label=int(cat.labels[i]), x=x, y=y, a=a, b=b,
                         theta_deg=np.degrees(theta), sky=sky,
                         flux=flux, flux_err=err, mask_corr=corr))
    return Table(rows)


def photometer_stars(data, xy, fwhm, r_ap=2.0, r_in=3.5, r_out=5.0, mask=None):
    """Circular aperture photometry (r_ap*FWHM) of stars at positions `xy`."""
    xy = np.atleast_2d(xy)
    ap = CircularAperture(xy, r=r_ap * fwhm)
    ann = CircularAnnulus(xy, r_in=r_in * fwhm, r_out=r_out * fwhm)
    st = ApertureStats(data, ann, mask=mask, sigma_clip=SigmaClip(3.0))
    ph = aperture_photometry(data, ap, mask=mask)
    flux = ph['aperture_sum'].value - st.median * ap.area
    err = st.std * np.sqrt(ap.area)
    return flux, err


# --------------------------------------------------------------------------
# 6. Reference stars -> zero point
# --------------------------------------------------------------------------
def load_reference(path):
    ref = Table.read(path, format='ascii.csv')
    if 'mag_r' not in ref.colnames:
        raise ValueError("Reference file needs a 'mag_r' column.")
    return ref


def match_reference_stars(cat, star_idx, ref, wcs=None, max_sep=3.0):
    """Match reference stars to detected stars; return (cat indices, mags)."""
    xs = np.asarray(cat.x_centroid)[star_idx]
    ys = np.asarray(cat.y_centroid)[star_idx]
    if {'ra', 'dec'} <= set(ref.colnames):
        if wcs is None or not wcs.has_celestial:
            raise ValueError("RA/Dec references need a WCS in the FITS header.")
        c_ref = SkyCoord(ref['ra'], ref['dec'], unit='deg')
        c_det = wcs.pixel_to_world(xs, ys)
        idx, sep, _ = c_ref.match_to_catalog_sky(c_det)
        pix_scale = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
        ok = sep.arcsec < max_sep * pix_scale
    else:
        from scipy.spatial import cKDTree
        d, idx = cKDTree(np.c_[xs, ys]).query(np.c_[ref['x'], ref['y']])
        ok = d < max_sep
    return star_idx[idx[ok]], np.asarray(ref['mag_r'])[ok]


def zero_point(star_flux, star_mags, exptime, sigma=2.5):
    """Sigma-clipped median zero point: ZP = m + 2.5 log10(F/t)."""
    good = star_flux > 0
    zps = star_mags[good] + 2.5 * np.log10(star_flux[good] / exptime)
    if zps.size == 0:
        raise RuntimeError("No usable reference stars for the zero point.")
    mean, med, std = sigma_clipped_stats(zps, sigma=sigma)
    n = np.sum(np.abs(zps - med) < sigma * max(std, 1e-9))
    return med, std / np.sqrt(max(n, 1)), zps


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(data, refs, exptime=1.0, wcs=None, nsigma=3.0, npixels=15,
        saturation=None, ext_factor=1.5, verbose=True):
    # pass 1: background with a rough source mask, then detection
    _, _, cat0 = None, None, None
    data = np.nan_to_num(data.astype(float), nan=np.nanmedian(data))
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    rough = data > med + 3 * std
    from scipy.ndimage import binary_dilation
    rough = binary_dilation(rough, iterations=5)
    sub, _ = subtract_background(data, source_mask=rough)

    segm, conv, cat = detect_objects(sub, nsigma=nsigma, npixels=npixels)
    is_ext, psf_fwhm = classify_sources(cat, ext_factor=ext_factor)
    ext_idx = np.where(is_ext)[0]
    star_idx = np.where(~is_ext & np.isfinite(cat.fwhm.value))[0]
    if verbose:
        print(f"Detected {len(cat)} sources: {len(star_idx)} stars, "
              f"{len(ext_idx)} extended. PSF FWHM ~ {psf_fwhm:.2f} px")

    star_mask = build_star_mask(sub.shape, cat, star_idx, psf_fwhm,
                                saturation=saturation)
    # Exclude other detected sources (not the galaxy's own) from sky annuli
    all_seg_mask = segm.data > 0
    mask_gal = star_mask  # stars masked inside galaxy apertures
    # sky annuli: mask stars and all other detections
    gal = photometer_galaxies(sub, cat, ext_idx, mask_gal,
                              sky_mask=star_mask | all_seg_mask)

    # zero point from matched reference stars (unmasked photometry)
    match_idx, ref_mags = match_reference_stars(cat, star_idx, refs, wcs)
    if len(match_idx) == 0:
        raise RuntimeError("No reference stars matched detected stars.")
    xy = np.c_[np.asarray(cat.x_centroid)[match_idx],
               np.asarray(cat.y_centroid)[match_idx]]
    # keep neighbouring stars' contamination out of the sky annulus
    sf, sfe = photometer_stars(sub, xy, psf_fwhm, mask=all_seg_mask & ~_own_mask(
        segm, cat, match_idx, sub.shape))
    zp, zp_err, zps = zero_point(sf, ref_mags, exptime)
    if verbose:
        print(f"{len(match_idx)} reference stars -> ZP = {zp:.3f} +/- {zp_err:.3f} "
              f"(individual: {np.round(zps, 3)})")

    f = gal['flux'] / exptime
    with np.errstate(divide='ignore', invalid='ignore'):
        gal['mag_r'] = np.where(f > 0, zp - 2.5 * np.log10(f), np.nan)
        merr = 1.0857 * gal['flux_err'] / gal['flux']
        gal['mag_r_err'] = np.where(f > 0, np.hypot(merr, zp_err), np.nan)
    for c in ('x', 'y', 'a', 'b', 'theta_deg', 'sky', 'flux', 'flux_err',
              'mask_corr', 'mag_r', 'mag_r_err'):
        gal[c].info.format = '.3f'
    if wcs is not None and wcs.has_celestial and len(gal):
        sky = wcs.pixel_to_world(gal['x'], gal['y'])
        gal['ra'], gal['dec'] = sky.ra.deg, sky.dec.deg
    return dict(galaxies=gal, zp=zp, zp_err=zp_err, psf_fwhm=psf_fwhm,
                image=sub, segm=segm, star_mask=star_mask,
                ref_xy=xy)


def _own_mask(segm, cat, idx, shape):
    """Pixels belonging to the given sources' own segments."""
    m = np.zeros(shape, bool)
    for i in idx:
        m |= segm.data == cat.labels[i]
    return m


def plot_result(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval
    img = res['image']
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, origin='lower', cmap='gray',
              norm=ImageNormalize(img, interval=ZScaleInterval(), stretch=SqrtStretch()))
    ax.contour(res['star_mask'], levels=[0.5], colors='cyan', linewidths=0.6)
    ax.plot(*res['ref_xy'].T, 'o', mfc='none', mec='yellow', ms=12, label='reference stars')
    for g in res['galaxies']:
        EllipticalAperture((g['x'], g['y']), g['a'], g['b'],
                           theta=np.radians(g['theta_deg'])).plot(ax=ax, color='red')
        ax.text(g['x'], g['y'] + g['a'] + 5, f"r={g['mag_r']:.2f}", color='red')
    ax.legend(loc='upper right')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def synthetic_field(n=800, seed=1, zp=25.0, exptime=60.0):
    """Fake field: Gaussian-PSF stars + Sersic galaxies + noise; truth mags."""
    from astropy.modeling.models import Gaussian2D, Sersic2D
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    img = np.full((n, n), 100.0)
    sig = 2.0
    stars = []
    for _ in range(25):
        x, y = rng.uniform(30, n - 30, 2)
        m = rng.uniform(14.5, 19)
        f = exptime * 10 ** (-0.4 * (m - zp))
        img += f / (2 * np.pi * sig ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))
        stars.append((x, y, m))
    gals = []
    for (x, y, m, re, ell, th) in [(200, 250, 16.5, 12, 0.3, 0.5),
                                   (550, 200, 17.0, 10, 0.5, 1.2),
                                   (400, 600, 16.0, 15, 0.2, 2.0)]:
        f = exptime * 10 ** (-0.4 * (m - zp))
        mod = Sersic2D(1, re, 1.0, x, y, ellip=ell, theta=th)
        prof = mod(xx, yy)
        img += prof / prof.sum() * f
        gals.append((x, y, m))
    img = rng.poisson(img) + rng.normal(0, 5, img.shape)
    return img, stars, gals, zp, exptime


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', nargs='?')
    p.add_argument('--refstars')
    p.add_argument('--exptime', type=float, default=1.0)
    p.add_argument('--ext', type=int, default=0, help='FITS extension')
    p.add_argument('--nsigma', type=float, default=3.0)
    p.add_argument('--npixels', type=int, default=15)
    p.add_argument('--saturation', type=float, default=None)
    p.add_argument('--out', default='galaxy_photometry.ecsv')
    p.add_argument('--plot', default=None)
    p.add_argument('--demo', action='store_true')
    a = p.parse_args()

    if a.demo:
        img, stars, gals, zp_true, exptime = synthetic_field()
        refs = Table(rows=[(x, y, m) for x, y, m in stars[:6]], names=('x', 'y', 'mag_r'))
        res = run(img, refs, exptime=exptime, saturation=a.saturation)
        print(f"True ZP = {zp_true}")
        print("True galaxy mags:", [g[2] for g in gals])
    else:
        if not (a.image and a.refstars):
            p.error('image and --refstars are required (or use --demo)')
        with fits.open(a.image) as hdul:
            data = hdul[a.ext].data
            hdr = hdul[a.ext].header
        try:
            wcs = WCS(hdr)
        except Exception:
            wcs = None
        exptime = a.exptime if a.exptime != 1.0 else float(hdr.get('EXPTIME', 1.0))
        res = run(data, load_reference(a.refstars), exptime=exptime, wcs=wcs,
                  nsigma=a.nsigma, npixels=a.npixels, saturation=a.saturation)
    res['galaxies'].write(a.out, overwrite=True)
    res['galaxies'].pprint_all()
    if a.plot or a.demo:
        plot_result(res, a.plot or 'galaxy_photometry_demo.png')


if __name__ == '__main__':
    main()
