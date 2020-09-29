# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:08:16 2020

@author: torres
"""

import numpy as np

def select_below_day(dem_nc,mask_nc,var1,elev):
    time_n,lat_n,lon_n = var1.shape
    for t in range(time_n):
        for i in range(lat_n):
            for j in range(lon_n):
                if dem_nc[i,j] <= 5000 and mask_nc[i,j] == 1:
                    var1[t,i,j] = var1[t,i,j]
                else:
                    var1[t,i,j] = np.nan
    return var1
