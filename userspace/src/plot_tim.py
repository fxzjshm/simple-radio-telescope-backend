#!/usr/bin/env python3
################################################################################
# Copyright (c) 2022 fxzjshm
# This software is licensed under Mulan PubL v2.
# You can use this software according to the terms and conditions of the Mulan PubL v2.
# You may obtain a copy of Mulan PubL v2 at:
#          http://license.coscl.org.cn/MulanPubL-2.0
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PubL v2 for more details.
################################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

def main():
    '''
    simple helper script to plot some data generated by the main program simple-radio-telescope-backend
    1) plot .tim file, i.e. time series, data_type = float32 or float64 depending on config
    2) plot (part of) recorded raw baseband data, data_type = int8, uint8 or something else depending on config.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("data_type")
    parser.add_argument("--size_limit", required=False, default=-1)
    args = parser.parse_args()

    time_series = np.fromfile(args.file_path, dtype=args.data_type, count=int(args.size_limit))
    mpl.rcParams['agg.path.chunksize'] = 10000
    plt.plot(time_series)
    plt.show()

if __name__ == '__main__':
    main()
