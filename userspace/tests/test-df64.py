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
import matplotlib.pyplot as plt

'''
This helper script reads binary file written by test-df64.cpp
to investigate the difference.
'''

def main():
    dedisp_double = np.fromfile("dedisp_double.bin", dtype=np.float32)
    dedisp_dsmath = np.fromfile("dedisp_dsmath.bin", dtype=np.float32)
    #plt.plot(dedisp_double)
    #plt.plot(dedisp_dsmath)
    plt.plot(dedisp_double - dedisp_dsmath)
    plt.show()


if __name__ == '__main__':
    main()
