#!/usr/bin/python
# -*- coding:utf-8 -*-

import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import reduce


if __name__ == "__main__":
    n = 100
    x = np.linspace(0, 1, 100,endpoint=True)
    print(x)
    y = 1/2*np.log((1-x)/x)
    mpl.rcParams[u'font.sans-serif'] = u'SimHei'
    mpl.rcParams[u'axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(x, y, 'ro-', lw=2)
    plt.xlim(0,1)
    # plt.ylim(-3, 3)
    plt.grid(b=True)
    plt.show()
