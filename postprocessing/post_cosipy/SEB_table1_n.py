"""
This script creates the figure the energy
balance in its distribution version 

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
from my_fun.select_below_n import select_below_n
from my_fun.select_season import select_dry, select_wet
#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
#filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = '../../data/output/Peru_out_50m_1_20160901-20170831.nc'
dir_graphics = 'out'
name_f       = 'table1.csv'

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

var1    = ds['G']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = select_dry(var1_m)
var1_w  = select_wet(var1_m)
table_SEB[0,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[0,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[0,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['G']
var1_d  = var1.resample(time = 'd').mean('time')
var2    = ds['ALBEDO']
var2_d  = var2.resample(time = 'd').mean('time')
var3_d  = var2.resample(time = 'd').mean('time')
time_d,x,y = var3_d.shape
for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3_d[t,i,j] = var1_d[t,i,j].values*(1-var2_d[t,i,j].values)

var1_m  = var3_d.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[1,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[1,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[1,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['LWin']
var1_d  = var1.resample(time = 'd').mean('time')
var2    = ds['LWout']
var2_d  = var2.resample(time = 'd').mean('time')
var3_d  = var2.resample(time = 'd').mean('time')
time_d,x,y = var3_d.shape
for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3_d[t,i,j] = var1_d[t,i,j].values+var2_d[t,i,j].values

var1_m  = var3_d.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[2,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[2,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[2,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['H']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[3,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[3,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[3,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['LE']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[4,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[4,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[4,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['B']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[5,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[5,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[5,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['ME']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[6,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[6,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[6,2] = np.round(np.nanmean(var1_w),decimals=0)


var1    = ds['RAIN']
var1_m  = var1.resample(time = 'm').sum('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.sum(select_dry(var1_m),axis=0)
var1_w  = np.sum(select_wet(var1_m),axis=0)
var1_m  = np.sum((var1_m),axis=0)

rain_m  = var1_m
rain_d  = var1_d
rain_w  = var1_w

var1    = ds['SNOWFALL']
var1_m  = var1.resample(time = 'm').sum('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.sum(select_dry(var1_m),axis=0)
var1_w  = np.sum(select_wet(var1_m),axis=0)
var1_m  = np.sum((var1_m),axis=0)

snow_m  = var1_m
snow_d  = var1_d
snow_w  = var1_w

TP_m = rain_m + snow_m
TP_d = rain_d + snow_d
TP_w = rain_w + snow_w

table_SEB[7,0] = np.round(np.nanmean(TP_m),decimals=2)
table_SEB[7,1] = np.round(np.nanmean(TP_d),decimals=2)
table_SEB[7,2] = np.round(np.nanmean(TP_w),decimals=2)

Pl_m = (np.nanmean(rain_m)/np.nanmean(TP_m))*100 
Pl_d = (np.nanmean(rain_d)/np.nanmean(TP_d))*100 
Pl_w = (np.nanmean(rain_w)/np.nanmean(TP_w))*100 

table_SEB[8,0] = np.round(Pl_m,decimals=0)
table_SEB[8,1] = np.round(Pl_d,decimals=0)
table_SEB[8,2] = np.round(Pl_w,decimals=0)

var1    = ds['T2']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values-273.15
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[9,0] = np.round(np.nanmean(var1_m),decimals=2)
table_SEB[9,1] = np.round(np.nanmean(var1_d),decimals=2)
table_SEB[9,2] = np.round(np.nanmean(var1_w),decimals=2)

var1    = ds['RH2']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[10,0] = np.round(np.nanmean(var1_m),decimals=0)
table_SEB[10,1] = np.round(np.nanmean(var1_d),decimals=0)
table_SEB[10,2] = np.round(np.nanmean(var1_w),decimals=0)

var1    = ds['U2']
var1_m  = var1.resample(time = 'm').mean('time')
var1_m  = var1_m.values
var1_m  = select_below_n(dem_nc,mask_nc,var1_m,5000)
var1_d  = np.nanmean(select_dry(var1_m),axis=0)
var1_w  = np.nanmean(select_wet(var1_m),axis=0)
var1_m  = np.nanmean((var1_m),axis=0)
table_SEB[11,0] = np.round(np.nanmean(var1_m),decimals=2)
table_SEB[11,1] = np.round(np.nanmean(var1_d),decimals=2)
table_SEB[11,2] = np.round(np.nanmean(var1_w),decimals=2)

pd.DataFrame(table_SEB).to_csv(dir_graphics+"/"+name_f,sep='\t', float_format='%.3f')
