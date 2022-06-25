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

# Usage: see `./filterbank-reader.py --help

import sigpyproc
import sys
import seaborn
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description="Dedisperse filterbank, and write time series or show heatmap")
    parser.add_argument("input", type=str, help="input .fil file")
    parser.add_argument("output", type=str, help="output file path, will show plot if invalid", default="")
    parser.add_argument("offset", type=int, help="read offset", default=0)
    parser.add_argument("nsamps", type=int, help="number of samples to read")
    parser.add_argument("--dm", type=float, help="dedisperse measurement", default=0.0)
    parser.add_argument("--vmin", help="min value when draw plot", default=None)
    parser.add_argument("--vmax", help="max value when draw plot", default=None)
    parser.add_argument("--tim", help="write/plot tim instead of filterbank", action="store_true")

    args = parser.parse_args()

    reader = sigpyproc.Readers.FilReader(args.input)
    block = reader.readBlock(int(args.offset), int(args.nsamps), True)
    if args.dm > 0.0 :
        print("Dedispersing, dm = ", args.dm)
        block = block.dedisperse(args.dm)

    if args.tim:
        tim = block.get_tim()
        try:
            tim.toDat(args.output)
        except Exception:
            map_sum = plt.plot(tim)
            plt.show()
    else:
        blockT = block.T
        try:
            blockT.tofile(sys.argv[2])
        except Exception:
            map = seaborn.heatmap(blockT, vmin=args.vmin, vmax=args.vmax)
            plt.show()

if __name__ == '__main__':
    main()
