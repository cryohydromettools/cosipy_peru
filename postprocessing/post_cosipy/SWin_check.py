import xarray as xr
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import netCDF4
import matplotlib.pyplot as plt 

filename_nc  = '../../data/input/Peru/Peru_input_1.nc'
#filename_nc = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_graphics   = 'out'
name_fig     = 'SWin'

ds = xr.open_dataset(filename_nc)

ds

time_nc = ds['time'].values
lon_nc  = ds['lon'].values
lat_nc  = ds['lat'].values
mask_nc = ds['MASK'].values

x,y = np.shape(mask_nc)

# select incident shortwave radiation for Artesonraju Glacier
for t in range(31,42):
    
    time_nc1 = time_nc[t]

    SWin_g   = ds['G'][t,:,:].values
    SWin_g_m = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if mask_nc[i,j] == 1:
                SWin_g_m[i,j] = SWin_g[i,j]
            else:
                SWin_g_m[i,j] = np.nan
    
    fig, (ax0) = plt.subplots(figsize=(3.5,3))    
    im0 = ax0.contourf(lon_nc, lat_nc, SWin_g_m,20, cmap=plt.get_cmap('jet'))
    ax0.set_yticks(np.round(np.linspace(np.min(lat_nc), np.max(lat_nc), 5), decimals=2))
    ax0.xaxis.set_ticks_position('top')
    ax0.xaxis.set_tick_params(which='both', rotation=90)
    ax0.set_xticks(np.round(np.linspace(np.min(lon_nc), np.max(lon_nc), 5), decimals=2))
    ax0.yaxis.set_tick_params(which='both', rotation=90)
    ax0.set_ylabel('Lat (°)')
    ax0.set_xlabel('Lon (°)')
    ax0.text(-77.649, -8.975, str(time_nc1)[0:13])      
    ax0.text(-77.649, -8.977,'min ='+str(np.round(np.nanmin(SWin_g_m),decimals=0)))     
    ax0.text(-77.649, -8.979,'max ='+str(np.round(np.nanmax(SWin_g_m),decimals=0)))      
    ax0.xaxis.set_label_position('top')    
    fig.savefig(dir_graphics+'/'+name_fig+'_'+str(t)+'.png',dpi = 300, bbox_inches = 'tight', 
                 pad_inches = 0.1)

