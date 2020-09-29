# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:58:26 2020

@author: torres
"""
import numpy as np

def select_dry(var1):
    # dry season
    id_dry  = np.array((0,8,9,10,11))
    t_m,lat_n,lon_n = var1.shape
    v_dry    = np.zeros((len(id_dry),lat_n,lon_n)) 
    v_dry[v_dry == 0]= np.nan 
    for i in range(len(v_dry)):
        v_dry[i,:,:] = var1[id_dry[i],:,:]
    return v_dry

def select_wet(var1):
    # wet season
    id_wet  = np.array((1,2,3,4,5,6,7))
    t_m,lat_n,lon_n = var1.shape
    v_wet    = np.zeros((len(id_wet),lat_n,lon_n)) 
    v_wet[v_wet == 0]= np.nan 
    for i in range(len(v_wet)):
        v_wet[i,:,:] = var1[id_wet[i],:,:]
    return v_wet
