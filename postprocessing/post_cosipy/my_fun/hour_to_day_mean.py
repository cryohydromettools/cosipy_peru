# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:37:32 2020

@author: torres
"""
import numpy as np

def hour_to_day_mean(days_sim,MB_hour):
    t1   = 0
    t2   = 24
    t_m,lat_n,lon_n = MB_hour.shape
    ME_day = np.zeros((days_sim,lat_n,lon_n))
    for t in range((days_sim)):
        var_nc    = MB_hour[t1:t2,:,:]
        ME_day[t,:,:] = np.nanmean(var_nc,axis=0)
        t1 = t1+24
        t2 = t2+24
    return ME_day
