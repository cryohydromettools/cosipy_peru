# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:47:09 2020

@author: torres
"""
import numpy as np

def select_below(dem_nc,mask_nc,var1,elev):
    lat_n,lon_n = var1.shape    
    for i in range(lat_n):
        for j in range(lon_n):
            if dem_nc[i,j] <= 5000 and mask_nc[i,j] == 1:
                var1[i,j] = var1[i,j]
            else:
                var1[i,j] = np.nan
    return var1
