# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:53:49 2020

@author: torres
"""

import numpy as np

def day_to_month_sum(month_sim,MB_day):
    days_month = np.array([0,30,31,30,31,31,28,31,30,31,30,31,31]) # start sept
    t_m,lat_n,lon_n = MB_day.shape
    ME_month = np.zeros((month_sim,lat_n,lon_n))
    for t in range((month_sim-1)):
        t1 = days_month[t]
        t2 = days_month[t]+days_month[t+1]
        var_nc    = MB_day[t1:t2,:,:]
        ME_month[t,:,:] = np.nansum(var_nc,axis=0)
    return ME_month
