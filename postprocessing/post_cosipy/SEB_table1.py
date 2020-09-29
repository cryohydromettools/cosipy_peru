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
from my_fun.select_below import select_below
from my_fun.hour_to_day_mean import hour_to_day_mean
from my_fun.hour_to_day_sum import hour_to_day_sum
from my_fun.select_season import select_dry, select_wet

filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
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

var_t = create_date(time_nc)

# select dry season for Artesonraju Glacier
str_t = np.datetime64('2016-09-01')
end_t = np.datetime64('2016-09-30')
ind_1 = np.where(var_t == str_t)
ind_1 = int(ind_1[0])
ind_2 = np.where(var_t == end_t)
ind_2 = int(ind_2[0])+1
str_t = np.datetime64('2017-05-01')
end_t = np.datetime64('2017-08-31')
ind_3 = np.where(var_t == str_t)
ind_3 = int(ind_3[0])
ind_4 = np.where(var_t == end_t)
ind_4 = int(ind_4[0])+1

# select wet season for Artesonraju Glacier
str_t = np.datetime64('2016-10-01')
end_t = np.datetime64('2017-04-30')
ind_1_w = np.where(var_t == str_t)
ind_1_w = int(ind_1_w[0])
ind_2_w = np.where(var_t == end_t)
ind_2_w = int(ind_2_w[0])+1

days_num = int(len(time_nc)/24)

var1a   = (ds['G'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[0,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[0,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[0,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['G'][:,:,:].values)
var1    = hour_to_day_mean(days_num,var1a)
var1a   = (ds['ALBEDO'][:,:,:].values)
var2    = hour_to_day_mean(days_num,var1a)

time_d,x,y = var2.shape
var3    = np.zeros((time_d,x,y))
for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3[t,i,j] = var1[t,i,j] * (1-var2[t,i,j])

var1_d  = var3
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)
SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[1,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[1,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[1,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['LWin'][:,:,:].values)
var1    = hour_to_day_mean(days_num,var1a)
var1a   = (ds['LWout'][:,:,:].values)
var2    = hour_to_day_mean(days_num,var1a)

time_d,x,y = var2.shape
var3    = np.zeros((time_d,x,y))

for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3[t,i,j] = var1[t,i,j] + var2[t,i,j]

var1_d  = var3
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)
SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[2,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[2,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[2,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['H'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[3,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[3,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[3,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['LE'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[4,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[4,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[4,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['B'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[5,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[5,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[5,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['ME'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[6,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[6,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[6,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['RAIN'][:,:,:].values)
var1_d  = hour_to_day_sum(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])

SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

rain_m  = SWin_m
rain_d  = SWin_d
rain_w  = SWin_w

var1a   = (ds['SNOWFALL'][:,:,:].values)
var1_d  = hour_to_day_sum(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])

SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

snow_m  = SWin_m
snow_d  = SWin_d
snow_w  = SWin_w

TP_m    = np.zeros((len(lat_nc),len(lon_nc)))
TP_m[TP_m == 0] = np.nan
TP_d    = np.zeros((len(lat_nc),len(lon_nc)))
TP_d[TP_d == 0] = np.nan
TP_w    = np.zeros((len(lat_nc),len(lon_nc)))
TP_w[TP_w == 0] = np.nan

for i in range(len(lat_nc)):
    for j in range(len(lon_nc)):
            if mask_nc[i,j] == 1 and dem_nc[i,j] < 5000:
                TP_m[i,j] = snow_m[i,j] + rain_m[i,j]
                TP_d[i,j] = snow_d[i,j] + rain_d[i,j]
                TP_w[i,j] = snow_w[i,j] + rain_w[i,j]
            else:
                TP_m[i,j] = np.nan
                TP_d[i,j] = np.nan
                TP_w[i,j] = np.nan

table_SEB[7,0] = np.round(np.nanmean(TP_m),decimals=2)
table_SEB[7,1] = np.round(np.nanmean(TP_d),decimals=2)
table_SEB[7,2] = np.round(np.nanmean(TP_w),decimals=2)

Pl_m = (np.nansum(rain_m)/np.nansum(TP_m))*100 
Pl_d = (np.nansum(rain_d)/np.nansum(TP_d))*100 
Pl_w = (np.nansum(rain_w)/np.nansum(TP_w))*100 

table_SEB[8,0] = np.round(Pl_m,decimals=0)
table_SEB[8,1] = np.round(Pl_d,decimals=0)
table_SEB[8,2] = np.round(Pl_w,decimals=0)

var1a   = (ds['T2'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])

SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[9,0] = np.round(np.nanmean(SWin_m)-273.15,decimals=2)
table_SEB[9,1] = np.round(np.nanmean(SWin_d)-273.15,decimals=2)
table_SEB[9,2] = np.round(np.nanmean(SWin_w)-273.15,decimals=2)

var1a   = (ds['RH2'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])

SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)

SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)

table_SEB[10,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[10,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[10,2] = np.round(np.nanmean(SWin_w),decimals=0)

var1a   = (ds['U2'][:,:,:].values)
var1_d  = hour_to_day_mean(days_num,var1a)
MB_d1   = (var1_d[ind_1:ind_2,:,:])
MB_d2   = (var1_d[ind_3:ind_4,:,:])
MB_d    = (np.concatenate((MB_d1,MB_d2), axis=0))
MB_w    = (var1_d[ind_1_w:ind_2_w,:,:])
SWin_m  = np.nanmean(var1_d, axis=0)
SWin_d  = np.nanmean(MB_d, axis=0)
SWin_w  = np.nanmean(MB_w, axis=0)
SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_d  = select_below(dem_nc,mask_nc,SWin_d,5000)
SWin_w  = select_below(dem_nc,mask_nc,SWin_w,5000)
table_SEB[11,0] = np.round(np.nanmean(SWin_m),decimals=2)
table_SEB[11,1] = np.round(np.nanmean(SWin_d),decimals=2)
table_SEB[11,2] = np.round(np.nanmean(SWin_w),decimals=2)

pd.DataFrame(table_SEB).to_csv(dir_graphics+"/"+name_f,sep='\t', float_format='%.3f')
