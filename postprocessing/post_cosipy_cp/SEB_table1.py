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
from my_fun.select_season import select_dry, select_wet

filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'out/Peru_output_1_new_20160901-20170831.nc'
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
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')

SWin_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
SWin_m  = select_below(dem_nc,mask_nc,SWin_m,5000)
SWin_m  = SWin_m.values
SWin_d  = select_dry(SWin_m)
SWin_w  = select_wet(SWin_m)
table_SEB[0,0] = np.round(np.nanmean(SWin_m),decimals=0)
table_SEB[0,1] = np.round(np.nanmean(SWin_d),decimals=0)
table_SEB[0,2] = np.round(np.nanmean(SWin_w),decimals=0)

var2    = ds['ALBEDO']
var2_d  = var2.resample(freq = 'd', dim = 'time', how = 'mean')
var3_d  = var2.resample(freq = 'd', dim = 'time', how = 'mean')
time_d,x,y = var3_d.shape

for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3_d[t,i,j] = var1_d[t,i,j].values*(1-var2_d[t,i,j].values)

SWnet_m  = var3_d.resample(freq = 'm', dim = 'time', how = 'mean')
SWnet_m  = select_below(dem_nc,mask_nc,SWnet_m,5000)
SWnet_m  = SWnet_m.values
SWnet_d  = select_dry(SWnet_m)
SWnet_w  = select_wet(SWnet_m)

SWnet_m  = np.nanmean(SWnet_m,axis=0)
SWnet_d  = np.nanmean(SWnet_d,axis=0)
SWnet_w  = np.nanmean(SWnet_w,axis=0)


table_SEB[1,0] = np.round(np.nanmean(SWnet_m),decimals=0)
table_SEB[1,1] = np.round(np.nanmean(SWnet_d),decimals=0)
table_SEB[1,2] = np.round(np.nanmean(SWnet_w),decimals=0)

var1    = ds['LWin']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var2    = ds['LWout']
var2_d  = var2.resample(freq = 'd', dim = 'time', how = 'mean')
var2_m  = var2_d.resample(freq = 'm', dim = 'time', how = 'mean')
var3_m  = var2_d.resample(freq = 'm', dim = 'time', how = 'mean')

time_d,x,y = var3_m.shape

for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3_m[t,i,j] = var1_m[t,i,j].values + var2_m[t,i,j].values

LWnet_m  = select_below(dem_nc,mask_nc,var3_m,5000)
LWnet_m  = LWnet_m.values
LWnet_d  = select_dry(LWnet_m)
LWnet_w  = select_wet(LWnet_m)

LWnet_m  = np.nanmean(LWnet_m,axis=0)
LWnet_d  = np.nanmean(LWnet_d,axis=0)
LWnet_w  = np.nanmean(LWnet_w,axis=0)

table_SEB[2,0] = np.round(np.nanmean(LWnet_m),decimals=0)
table_SEB[2,1] = np.round(np.nanmean(LWnet_d),decimals=0)
table_SEB[2,2] = np.round(np.nanmean(LWnet_w),decimals=0)

var1    = ds['H']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
H_m  = var1_m.values
H_d  = select_dry(H_m)
H_w  = select_wet(H_m)
table_SEB[3,0] = np.round(np.nanmean(H_m),decimals=0)
table_SEB[3,1] = np.round(np.nanmean(H_d),decimals=0)
table_SEB[3,2] = np.round(np.nanmean(H_w),decimals=0)

var1    = ds['LE']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
LE_m  = var1_m.values
LE_d  = select_dry(LE_m)
LE_w  = select_wet(LE_m)
LE_m  = np.nanmean(LE_m,axis=0)
LE_d  = np.nanmean(LE_d,axis=0)
LE_w  = np.nanmean(LE_w,axis=0)

table_SEB[4,0] = np.round(np.nanmean(LE_m),decimals=0)
table_SEB[4,1] = np.round(np.nanmean(LE_d),decimals=0)
table_SEB[4,2] = np.round(np.nanmean(LE_w),decimals=0)

var1    = ds['B']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
LE_m  = var1_m.values
LE_d  = select_dry(LE_m)
LE_w  = select_wet(LE_m)
table_SEB[5,0] = np.round(np.nanmean(LE_m),decimals=0)
table_SEB[5,1] = np.round(np.nanmean(LE_d),decimals=0)
table_SEB[5,2] = np.round(np.nanmean(LE_w),decimals=0)

var1    = ds['ME']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
ME_m  = var1_m.values
ME_d  = select_dry(ME_m)
ME_w  = select_wet(ME_m)
ME_m  = np.nanmean(ME_m,axis=0)
ME_d  = np.nanmean(ME_d,axis=0)
ME_w  = np.nanmean(ME_w,axis=0)

table_SEB[6,0] = np.round(np.nanmean(ME_m),decimals=0)
table_SEB[6,1] = np.round(np.nanmean(ME_d),decimals=0)
table_SEB[6,2] = np.round(np.nanmean(ME_w),decimals=0)

var1    = ds['RAIN']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'sum')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'sum')

rain_m  = select_below(dem_nc,mask_nc,var1_m,5000)
rain_m  = rain_m.values
rain_d  = select_dry(rain_m)
rain_w  = select_wet(rain_m)

var2    = ds['SNOWFALL']
var2_d  = var2.resample(freq = 'd', dim = 'time', how = 'sum')
var2_m  = var2_d.resample(freq = 'm', dim = 'time', how = 'sum')
var3_m  = var2_d.resample(freq = 'm', dim = 'time', how = 'sum')

time_d,x,y = var3_m.shape

for t in range(time_d):
    for i in range(lat_n):
        for j in range(lon_n):
                if mask_nc[i,j] == 1:
                    var3_m[t,i,j] = var1_m[t,i,j].values + var2_m[t,i,j].values

TP_m  = select_below(dem_nc,mask_nc,var3_m,5000)
TP_m  = TP_m.values
TP_d  = select_dry(TP_m)
TP_w  = select_wet(TP_m)
TP_m  = np.nansum(TP_m,axis=0)
TP_d  = np.nansum(TP_d,axis=0)
TP_w  = np.nansum(TP_w,axis=0)

table_SEB[7,0] = np.round(np.nanmean(TP_m),decimals=2)
table_SEB[7,1] = np.round(np.nanmean(TP_d),decimals=2)
table_SEB[7,2] = np.round(np.nanmean(TP_w),decimals=2)

rain_m  = rain_m.values
rain_d  = select_dry(rain_m)
rain_w  = select_wet(rain_m)
rain_m  = np.nansum(rain_m,axis=0)
rain_d  = np.nansum(rain_d,axis=0)
rain_w  = np.nansum(rain_w,axis=0)

Pl_m = (np.nanmean(rain_m)/np.nanmean(TP_m))*100 
Pl_d = (np.nanmean(rain_d)/np.nanmean(TP_d))*100 
Pl_w = (np.nanmean(rain_w)/np.nanmean(TP_w))*100 

table_SEB[8,0] = np.round(Pl_m,decimals=0)
table_SEB[8,1] = np.round(Pl_d,decimals=0)
table_SEB[8,2] = np.round(Pl_w,decimals=0)

var1    = ds['T2']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
T2_m  = var1_m.values
T2_d  = select_dry(T2_m)
T2_w  = select_wet(T2_m)
T2_m  = np.nanmean(T2_m,axis=0)
T2_d  = np.nanmean(T2_d,axis=0)
T2_w  = np.nanmean(T2_w,axis=0)

table_SEB[9,0] = np.round(np.nanmean(T2_m)-272.15,decimals=2)
table_SEB[9,1] = np.round(np.nanmean(T2_d)-272.15,decimals=2)
table_SEB[9,2] = np.round(np.nanmean(T2_w)-272.15,decimals=2)

var1    = ds['RH2']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
RH2_m  = var1_m.values
RH2_d  = select_dry(RH2_m)
RH2_w  = select_wet(RH2_m)
RH2_m  = np.nanmean(RH2_m,axis=0)
RH2_d  = np.nanmean(RH2_d,axis=0)
RH2_w  = np.nanmean(RH2_w,axis=0)
table_SEB[10,0] = np.round(np.nanmean(RH2_m),decimals=0)
table_SEB[10,1] = np.round(np.nanmean(RH2_d),decimals=0)
table_SEB[10,2] = np.round(np.nanmean(RH2_w),decimals=0)

var1    = ds['U2']
var1_d  = var1.resample(freq = 'd', dim = 'time', how = 'mean')
var1_m  = var1_d.resample(freq = 'm', dim = 'time', how = 'mean')
var1_m  = select_below(dem_nc,mask_nc,var1_m,5000)
U2_m  = var1_m.values
U2_d  = select_dry(U2_m)
U2_w  = select_wet(U2_m)
U2_m  = np.nanmean(U2_m,axis=0)
U2_d  = np.nanmean(U2_d,axis=0)
U2_w  = np.nanmean(U2_w,axis=0)
table_SEB[11,0] = np.round(np.nanmean(U2_m),decimals=2)
table_SEB[11,1] = np.round(np.nanmean(U2_d),decimals=2)
table_SEB[11,2] = np.round(np.nanmean(U2_w),decimals=2)

pd.DataFrame(table_SEB).to_csv(dir_graphics+"/"+name_f,sep='\t', float_format='%.3f')
