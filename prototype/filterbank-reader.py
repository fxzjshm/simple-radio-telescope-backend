#!/usr/bin/env python3

# Usage: ./filterbank-reader.py <input .fil path> <output .tim path> <start offset> <nsamps to read> [DM to dedisperse]
#        will show plot of output path is invalid

import sigpyproc
import sys
import seaborn
import matplotlib.pyplot as plt

reader = sigpyproc.Readers.FilReader(sys.argv[1])
block = reader.readBlock(int(sys.argv[3]), int(sys.argv[4]), True)
if len(sys.argv) > 5 :
    print(float(sys.argv[5]))
    block = block.dedisperse(float(sys.argv[5]))
blockT = block.T
try:
    #f = open(sys.argv[2], 'w')
    #writer = csv.writer(f, delimiter=' ')
    #writer.writerows(blockT)
    #blockT.tofile(sys.argv[2])
    block.get_tim().toDat(sys.argv[2])
except Exception:
    #map_sum = plt.plot(block.get_tim())
    #plt.show()
    map = seaborn.heatmap(blockT)
    plt.show()
    
