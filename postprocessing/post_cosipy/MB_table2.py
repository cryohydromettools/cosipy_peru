# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:00:59 2020

@author: torres
"""
import xarray as xr
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import netCDF4
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from my_fun.create_date import create_date
from my_fun.nan_helper import nan_helper
from my_fun.select_below import select_below
from my_fun.select_season import select_dry, select_wet

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
#filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = '../../data/output/Peru_out_50m_1_20160901-20170831.nc'
dir_graphics = 'out'
name_f       = 'table2.csv'

ds = xr.open_dataset(filename_nc)

table_SEB = np.zeros((12,3)) 
table_SEB[table_SEB == 0] = np.nan

time_nc = ds['time'].values
mask_nc = ds['MASK'].values
lat_nc  = ds['lat'].values        
lon_nc  = ds['lon'].values
dem_nc   = ds['HGT'].values
lat_n = len(lat_nc)
lon_n = len(lon_nc)

var_t = create_date(time_nc)

table_SEB = np.zeros((15,3)) 
table_SEB[table_SEB == 0] = np.nan
# select dry season for Artesonraju Glacier
str_t = np.datetime64('2016-09-01T00:00:00.000000000')
end_t = np.datetime64('2016-09-30T23:00:00.000000000')
ind_1 = np.where(time_nc == str_t)
ind_1 = int(ind_1[0])
ind_2 = np.where(time_nc == end_t)
ind_2 = int(ind_2[0])
str_t = np.datetime64('2017-05-01T00:00:00.000000000')
end_t = np.datetime64('2017-08-31T23:00:00.000000000')
ind_3 = np.where(time_nc == str_t)
ind_3 = int(ind_3[0])
ind_4 = np.where(time_nc == end_t)
ind_4 = int(ind_4[0])

# select wet season for Artesonraju Glacier
str_t = np.datetime64('2016-10-01T00:00:00.000000000')
end_t = np.datetime64('2017-04-30T23:00:00.000000000')
ind_1_w = np.where(time_nc == str_t)
ind_1_w = int(ind_1_w[0])
ind_2_w = np.where(time_nc == end_t)
ind_2_w = int(ind_2_w[0])


MB_m  = np.nansum(ds['MB'][:,:,:].values,axis=0)
MB_d1  = (ds['MB'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['MB'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['MB'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[0,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[0,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[0,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['surfMB'][:,:,:].values,axis=0)
MB_d1  = (ds['surfMB'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['surfMB'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['surfMB'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[1,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[1,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[1,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['intMB'][:,:,:].values,axis=0)
MB_d1  = (ds['intMB'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['intMB'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['intMB'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[2,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[2,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[2,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['SNOWFALL'][:,:,:].values,axis=0)
MB_d1  = (ds['SNOWFALL'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['SNOWFALL'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['SNOWFALL'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[3,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[3,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[3,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['DEPOSITION'][:,:,:].values,axis=0)
MB_d1  = (ds['DEPOSITION'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['DEPOSITION'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['DEPOSITION'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[4,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[4,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[4,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['surfM'][:,:,:].values,axis=0)
MB_d1  = (ds['surfM'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['surfM'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['surfM'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[5,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[5,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[5,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['SUBLIMATION'][:,:,:].values,axis=0)
MB_d1  = (ds['SUBLIMATION'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['SUBLIMATION'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['SUBLIMATION'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[6,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[6,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[6,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['EVAPORATION'][:,:,:].values,axis=0)
MB_d1  = (ds['EVAPORATION'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['EVAPORATION'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['EVAPORATION'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[7,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[7,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[7,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['REFREEZE'][:,:,:].values,axis=0)
MB_d1  = (ds['REFREEZE'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['REFREEZE'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['REFREEZE'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[8,0] = np.round(np.nanmean(MB_m),decimals=4)
table_SEB[8,1] = np.round(np.nanmean(MB_d),decimals=4)
table_SEB[8,2] = np.round(np.nanmean(MB_w),decimals=4)

MB_m  = np.nansum(ds['subM'][:,:,:].values,axis=0)
MB_d1  = (ds['subM'][ind_1:ind_2,:,:].values)
MB_d2  = (ds['subM'][ind_3:ind_4,:,:].values)
MB_d   = np.nansum(np.concatenate((MB_d1,MB_d2), axis=0),axis=0)
MB_w   = np.nansum(ds['subM'][ind_1_w:ind_2_w,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                MB_m[i,j] = MB_m[i,j]
                MB_d[i,j] = MB_d[i,j]
                MB_w[i,j] = MB_w[i,j]
            else:
                MB_m[i,j] = np.nan
                MB_d[i,j] = np.nan
                MB_w[i,j] = np.nan

table_SEB[9,0] = np.round(-np.nanmean(MB_m),decimals=4)
table_SEB[9,1] = np.round(-np.nanmean(MB_d),decimals=4)
table_SEB[9,2] = np.round(-np.nanmean(MB_w),decimals=4)

var_3  = np.nansum(ds['MB'][:,:,:].values,axis=0)

dem_nc   = ds['HGT'].values
point_g = np.sum(mask_nc[mask_nc==1])
elev_g  = np.zeros((int(point_g),1))
x,y = np.shape(mask_nc)
z1 = 0
for i in range(x):
    for j in range(y):
            if mask_nc[i,j] == 1:
                var1 = dem_nc[i,j]
                elev_g[z1,0] = var1
                z1 = z1+1

MB_he_t = np.zeros((len(elev_g),1))

z1 = 0
for i in range(x):
    for j in range(y):
        if mask_nc[i,j] == 1:
            MB_he_t[z1,0] = var_3[i,j]
            z1 = z1+1
vara1 = np.concatenate((elev_g, MB_he_t), axis=1)
vara2 = vara1[np.lexsort(([vara1[:, i] for i in range(vara1.shape[1]-1, -1, -1)]))]
MB_he_t = vara2[:,1]
MB_he_t[MB_he_t<=0.0] = np.nan
ind_nan = np.argwhere(np.isnan(MB_he_t))
index_sla = int(ind_nan[-1])
ela_t = vara2[index_sla ,0]
table_SEB[10,0] = np.round(ela_t,decimals=0)

var_3   = ds['SNOWHEIGHT'][:,:,:].values

z,x,y = np.shape(var_3)  

dem_nc   = ds['HGT'].values
point_g = np.sum(mask_nc[mask_nc==1])
elev_g  = np.zeros((int(point_g),1))


z1 = 0
for i in range(x):
    for j in range(y):
            if mask_nc[i,j] == 1:
                var1 = dem_nc[i,j]
                elev_g[z1,0] = var1
                z1 = z1+1
snow_he_t = np.zeros((len(elev_g),z))

for k in range(z):
    var_1 = var_3[k,:,:]
    var_2 = np.zeros((len(elev_g),1))
    z1 = 0
    for i in range(x):
        for j in range(y):
            if mask_nc[i,j] == 1:
                var_2[z1,0] = var_1[i,j]
                z1 = z1+1
    vara1 = np.concatenate((elev_g, var_2), axis=1)
    vara2 = vara1[np.lexsort(([vara1[:, i] for i in range(vara1.shape[1]-1, -1, -1)]))]
    snow_he_t[:,k] = vara2[:,1]

snow_he_t = np.mean(snow_he_t, axis=1)

snow_he_t[snow_he_t<0.1] = np.nan
ind_nan = np.argwhere(np.isnan(snow_he_t))
index_sla = int(ind_nan[-1])
sla_t = vara2[index_sla,0]

table_SEB[11,0] = np.round(sla_t,decimals=0)


var_1   = (ds['SNOWHEIGHT'][ind_1:ind_2,:,:].values)
var_2   = (ds['SNOWHEIGHT'][ind_3:ind_4,:,:].values)

var_3   = np.concatenate((var_1,var_2),axis=0)

z,x,y = np.shape(var_3)  

dem_nc   = ds['HGT'].values
point_g = np.sum(mask_nc[mask_nc==1])
elev_g  = np.zeros((int(point_g),1))

z1 = 0
for i in range(x):
    for j in range(y):
            if mask_nc[i,j] == 1:
                var1 = dem_nc[i,j]
                elev_g[z1,0] = var1
                z1 = z1+1
snow_he_d = np.zeros((len(elev_g),z))

for k in range(z):
    var_1 = var_3[k,:,:]
    var_2 = np.zeros((len(elev_g),1))
    z1 = 0
    for i in range(x):
        for j in range(y):
            if mask_nc[i,j] == 1:
                var_2[z1,0] = var_1[i,j]
                z1 = z1+1
    vara1 = np.concatenate((elev_g, var_2), axis=1)
    vara2 = vara1[np.lexsort(([vara1[:, i] for i in range(vara1.shape[1]-1, -1, -1)]))]
    snow_he_d[:,k] = vara2[:,1]

snow_he_d = np.mean(snow_he_d, axis=1)
snow_he_d[snow_he_d<0.1] = np.nan
ind_nan = np.argwhere(np.isnan(snow_he_d))
index_sla = int(ind_nan[-1])
sla_d = vara2[index_sla,0]

table_SEB[11,1] = np.round(sla_d,decimals=0)

var_3   = (ds['SNOWHEIGHT'][ind_1_w:ind_2_w,:,:].values)
z,x,y = np.shape(var_3)  

dem_nc   = ds['HGT'].values
point_g = np.sum(mask_nc[mask_nc==1])
elev_g  = np.zeros((int(point_g),1))

z1 = 0
for i in range(x):
    for j in range(y):
            if mask_nc[i,j] == 1:
                var1 = dem_nc[i,j]
                elev_g[z1,0] = var1
                z1 = z1+1
snow_he_w = np.zeros((len(elev_g),z))

for k in range(z):
    var_1 = var_3[k,:,:]
    var_2 = np.zeros((len(elev_g),1))
    z1 = 0
    for i in range(x):
        for j in range(y):
            if mask_nc[i,j] == 1:
                var_2[z1,0] = var_1[i,j]
                z1 = z1+1
    vara1 = np.concatenate((elev_g, var_2), axis=1)
    vara2 = vara1[np.lexsort(([vara1[:, i] for i in range(vara1.shape[1]-1, -1, -1)]))]
    snow_he_w[:,k] = vara2[:,1]
snow_he_w = np.mean(snow_he_w, axis=1)
snow_he_w[snow_he_w<0.1] = np.nan
ind_nan = np.argwhere(np.isnan(snow_he_w))
index_sla = int(ind_nan[-1])
sla_w = vara2[index_sla,0]

table_SEB[11,2] = np.round(sla_w,decimals=0)

var_pos_t  = np.nansum(ds['MB'][:,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                var_pos_t[i,j] = var_pos_t[i,j]
            else:
                var_pos_t[i,j] = np.nan

var_pos_t[var_pos_t <= 0]   = np.nan 
var_pos_t[var_pos_t >= 0] = 1
AAR_pos_t = np.sum(var_pos_t[var_pos_t==1])

var_neg_t  = np.nansum(ds['MB'][:,:,:].values,axis=0)
for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1:
                var_neg_t[i,j] = var_neg_t[i,j]
            else:
                var_neg_t[i,j] = np.nan

var_neg_t[var_neg_t > 0]   = np.nan 
var_neg_t[var_neg_t <= 0] = 1
AAR_neg_t = np.sum(var_neg_t[var_neg_t==1])

AAR_t = AAR_neg_t/AAR_pos_t
table_SEB[12,0] = np.round(AAR_t,decimals=4)


MB_pos_t   = np.nansum(ds['MB'][:,:,:].values,axis=0)

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1 and dem_nc[i,j] >= ela_t:
                MB_pos_t[i,j] = MB_pos_t[i,j]
            else:
                MB_pos_t[i,j] = np.nan

table_SEB[13,0] = np.nanmean(np.round(MB_pos_t,decimals=4))

MB_neg_t   = np.nansum(ds['MB'][:,:,:].values,axis=0)
for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1 and dem_nc[i,j] < ela_t:
                MB_neg_t[i,j] = MB_neg_t[i,j]
            else:
                MB_neg_t[i,j] = np.nan

table_SEB[14,0] = np.nanmean(np.round(MB_neg_t,decimals=4))
pd.DataFrame(table_SEB).to_csv(dir_graphics+"/"+name_f,sep='\t', float_format='%.3f')
