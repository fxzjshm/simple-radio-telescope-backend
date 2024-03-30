/********************************************************
 *                                                      *
 * Licensed under the Academic Free License version 3.0 *
 *                                                      *
 ********************************************************/

#ifndef __VCSBEAM_HPP__
#define __VCSBEAM_HPP__

// selected code from MWA CIRA Pulsars and Transients Group's vcsbeam
// https://github.com/CIRA-Pulsars-and-Transients-Group/vcsbeam/

#include <erfa.h>
#include <star/pal.h>

// clang-format off

/* Get angle constants from PAL, if available;
 * otherwise construct them from math.h
 */
#if __has_include(<star/palmac.h>)
#include <star/palmac.h>
#define PIBY2  PAL__DPIBY2
#define H2R    PAL__DH2R
#define R2H    PAL__DR2H
#define D2R    PAL__DD2R
#define R2D    PAL__DR2D
#else
#define PIBY2  (0.5*M_PI)
#define H2R    (M_PI/12.0)
#define R2H    (12.0/M_PI)
#define D2R    (M_PI/180.0)
#define R2D    (180.0/M_PI)
#endif

typedef struct beam_geom_t {
    double mean_ra;
    double mean_dec;
    double az;
    double el;
    double lmst;
    double fracmjd;
    double intmjd;
    double unit_N;
    double unit_E;
    double unit_H;
} beam_geom;

/**
 * Convert MJD to LST.
 *
 * @param[in]  mjd  The Modified Julian Date
 * @param[out] lst  The Local Sidereal Time
 *
 * @todo Consider removing mjd2lst(), since it consists only of a single
 *       call to a `pal` function.
 */
inline
void mjd2lst(double lon_rad, double mjd, double *lst)
{
    // Greenwich Mean Sidereal Time to LMST
    // east longitude in hours at the epoch of the MJD
    double lmst = palRanorm(palGmst(mjd) + lon_rad);

    *lst = lmst;
}

/**
 * Populates a `beam_geom` struct with pointing information derived from a
 * given set of RAs, Decs, and MJDs.
 *
 * @param[in]  ras_hours An array of RAs (in decimal hours)
 * @param[in]  decs_degs An array of Decs (in decimal degrees)
 * @param[in]  mjd       The Modified Julian Date
 * @param[in]  lon_rad   Longitude of reference in radian (replaced MWA_LATITUDE_RADIANS)
 * @param[out] bg        The struct containing various geometric quantities
 *
 * The quantities which are calculated are
 * | Quantity       | Description                    |
 * | -------------- | ------------------------------ |
 * | `bg->mean_ra`  | The mean RA of the pointing    |
 * | `bg->mean_dec` | The mean Dec of the pointing   |
 * | `bg->az`       | The azimuth of the pointing    |
 * | `bg->el`       | The elevation of the pointing  |
 * | `bg->lmst`     | The local mean sidereal time   |
 * | `bg->fracmjd`  | The fractional part of the MJD |
 * | `bg->intmjd`   | The integer part of the MJD    |
 * | `bg->unit_N`   | The normalised projection of the look-direction onto local North |
 * | `bg->unit_E`   | The normalised projection of the look-direction onto local East  |
 * | `bg->unit_H`   | The normalised projection of the look-direction onto local "Up"  |
 *
 * @todo Put the table describing the beam_geom struct where it belongs: with
 *       the documentation for the beam_geom struct!
 */
inline
void calc_beam_geom(
        double            ras_hours,
        double            decs_degs,
        double            mjd,
        double            lon_rad,
        beam_geom        *bg )
{
    // Calculate geometry of pointings

    double intmjd;
    double fracmjd;
    double lmst;
    double mean_ra, mean_dec, ha;
    double az, el;

    double unit_N;
    double unit_E;
    double unit_H;

    double pr = 0, pd = 0, px = 0, rv = 0, eq = 2000, ra_ap = 0, dec_ap = 0;

    /* get mjd */
    intmjd = floor(mjd);
    fracmjd = mjd - intmjd;

    /* get requested Az/El from command line */
    mjd2lst( lon_rad, mjd, &lmst );

    /* for the look direction <not the tile> */

    mean_ra = ras_hours * H2R;
    mean_dec = decs_degs * D2R;

    palMap(mean_ra, mean_dec, pr, pd, px, rv, eq, mjd, &ra_ap, &dec_ap);

    // Lets go mean to apparent precess from J2000.0 to EPOCH of date.

    ha = palRanorm( lmst - ra_ap ); // in radians

    /* now HA/Dec to Az/El */

    palDe2h( ha, dec_ap, lon_rad, &az, &el );
    // ^-- Returns "geographic azimuth" and "elevation" (see documentation)

    /* now we need the direction cosines */

    unit_N = cos(el) * cos(az);
    unit_E = cos(el) * sin(az);
    unit_H = sin(el);

    // Populate a structure with some of the calculated values
    bg->mean_ra  = mean_ra;
    bg->mean_dec = mean_dec;
    bg->az       = az;
    bg->el       = el;
    bg->lmst     = lmst;
    bg->fracmjd  = fracmjd;
    bg->intmjd   = intmjd;
    bg->unit_N   = unit_N;
    bg->unit_E   = unit_E;
    bg->unit_H   = unit_H;
}

// clang-format on

#endif  // __VCSBEAM_HPP__
