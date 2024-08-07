/*******************************************************************************
 * Copyright (c) 2024 fxzjshm
 * 21cma-make_beam is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_21CMA_MAKE_BEAM_GLOBAL_VARIABLES__
#define __SRTB_21CMA_MAKE_BEAM_GLOBAL_VARIABLES__

#include <map>
#include <string>
#include <vector>

#include "common.hpp"

namespace srtb::_21cma::make_beam {

// TODO: read data from files

/** @brief effective "cable" relative time delay, unit: m */
inline std::map<std::string, double> cable_delay_table = {
  {"E01", 0.0},
  {"E02", -1609.0019516398847},
  {"E03", 0.0},
  {"E04", 0.0},
  {"E05", 0.0},
  {"E06", 0.0},
  {"E07", -1023.0177985155273},
  {"E08", 0.0},
  {"E09", -840.8783668789913},
  {"E10", -815.72235239904},
  {"E11", -784.4699316488721},
  {"E12", 0.0},
  {"E13", 0.0},
  {"E14", 0.0},
  {"E15", 0.0},
  {"E16", 0.0},
  {"E17", -365.30118321728963},
  {"E18", 0.0},
  {"E19", 0.0},
  {"E20", 0.0},
  {"W01", 0.0},
  {"W02", 0.0},
  {"W03", 0.0},
  {"W04", 0.0},
  {"W05", -970.5450124752957},
  {"W06", -910.6849950097744},
  {"W07", 0.0},
  {"W08", 0.0},
  {"W09", 0.0},
  {"W10", 0.0},
  {"W11", -613.9348434775306},
  {"W12", 0.0},
  {"W13", -447.99893271895223},
  {"W14", 0.0},
  {"W15", 0.0},
  {"W16", -23.829690328980178},
  {"W17", 9.13509203866056},
  {"W18", 0.0},
  {"W19", 220.9935505397002},
  {"W20", 0.0}
};

inline std::vector<std::string> station_map = {
    "E17", "E01", "E18", "W11", "E07", "USRP", "W16", "W19", "W06", "E10", "W13", "W05", "E11", "E02", "E09", "W17",
};

/** @brief relative locations, axis: local E, N, H, unit: meter */
inline std::map<std::string, relative_location_t> antenna_location = {
    {"E01", {0.0, 0.0, 0.0}},
    {"E02", {39.85471149474458, -0.09673947607821343, 0.8881213036390821}},
    {"E03", {139.87175015651485, -0.25691064591578305, 2.298169395897535}},
    {"E04", {159.84233032104018, -0.2099301000620206, 2.7728399559850985}},
    {"E05", {239.8673730247173, -0.2629654063911705, 4.000638201473673}},
    {"E06", {419.89771940129117, -0.548689406997512, 8.20635578578795}},
    {"E07", {439.92195588025254, -0.5544450056765786, 7.945058574510345}},
    {"E08", {539.9066770233258, -0.5678391643180081, 8.932250363017696}},
    {"E09", {559.9369139550747, -0.589711836472951, 9.832135447743028}},
    {"E10", {579.8828039441659, -0.6184501737350229, 10.884281546060038}},
    {"E11", {599.6250955293539, -0.412526785759593, 11.156727884020908}},
    {"E12", {619.9440540039008, -0.7014530656895236, 11.448788207886237}},
    {"E13", {639.8461286065948, -0.7015937049989459, 11.92909397698666}},
    {"E14", {739.8337480736221, -0.8578917992457747, 14.83955933378036}},
    {"E15", {779.9401620666241, -0.9025017165731123, 15.821403336315647}},
    {"E16", {819.9001753219413, -0.9040438129694587, 17.606144044014865}},
    {"E17", {879.8780311825784, -1.008597327463586, 19.645175564990872}},
    {"E18", {1139.7747531134942, -1.218993455721696, 29.930782940455895}},
    {"E19", {1259.8044029734715, -1.5261544761469676, 35.818556589041435}},
    {"E20", {1279.7904117219891, -1.549639270324858, 36.2228907403784}},
    {"W01", {-180.06453891258727, 0.015221527023313447, -2.3460452161747547}},
    {"W02", {-200.0601541092521, 0.0708012067272842, -2.3310407124256365}},
    {"W03", {-320.05503182054645, 0.156434263873402, -4.157495800448114}},
    {"W04", {-580.1349831478652, 0.5770726491527611, -7.403576633484469}},
    {"W05", {-640.1502118877764, 0.31854737118389587, -9.61766122407638}},
    {"W06", {-680.503024820927, 0.415760005210452, -9.318385933463507}},
    {"W07", {-720.0986650569317, 0.5408149499275935, -9.723759530337725}},
    {"W08", {-820.0161692541958, 0.5217979762645887, -11.23579342752988}},
    {"W09", {-840.0994529615928, 0.44059339070101894, -11.613706799593942}},
    {"W10", {-860.1607497317985, 0.5210936590911915, -11.659881780235777}},
    {"W11", {-880.135637127194, 0.5135347345804646, -12.29245069296106}},
    {"W12", {-900.0849408972209, 0.4768057484545462, -13.081604478522914}},
    {"W13", {-920.09781817045, 0.47662753486135995, -13.32858667916627}},
    {"W14", {-1020.1442953950672, 0.4147778233019068, -14.337905616745445}},
    {"W15", {-1040.0968709028075, 0.4367734931858349, -14.641033426645794}},
    {"W16", {-1220.1585166574175, 0.5722016593868557, -16.44491880333128}},
    {"W17", {-1300.1076005076686, 0.5695812317207776, -17.449397725799734}},
    {"W18", {-1320.2114599075787, 0.6475705407747958, -17.831819251328852}},
    {"W19", {-1420.2141037934177, 0.6673708281755896, -18.447795381681548}},
    {"W20", {-1460.1462681369353, 0.5915001195593085, -19.248458733120096}},
};

/** @brief location of reference point, axis: longitude, latitude, height, unit: degree, degree, meter */
inline earth_location_t reference_point = {86.71737156, 42.92424534, 2598.13130186};

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_GLOBAL_VARIABLES__
