# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:23:27 2020

@author: torres
"""

import xarray as xr
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import netCDF4
import matplotlib.pyplot as plt 
import matplotlib as mpl
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from my_fun.nan_helper import nan_helper
from my_fun.reverse_colourmap import reverse_colourmap
from scipy.signal import savgol_filter

table_SEB = np.zeros((6)) 
table_SEB[table_SEB == 0] = np.nan

dir_graphics = 'out' 
name_f       = 'add_sim.csv'

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_output   = 'out'
name_table1  = 'stat_MB'
name_fig     = 'MB_esp'
ds = xr.open_dataset(filename_nc)
MB_all  = np.sum(ds['MB'].values,axis=0)

table_SEB[0] = np.nanmean(MB_all)

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_output   = 'out'
name_table1  = 'stat_MB'
name_fig     = 'MB_esp'
ds = xr.open_dataset(filename_nc)
MB_all  = np.sum(ds['MB'].values,axis=0)

table_SEB[1] = np.nanmean(MB_all)

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_output   = 'out'
name_table1  = 'stat_MB'
name_fig     = 'MB_esp'
ds = xr.open_dataset(filename_nc)
MB_all  = np.sum(ds['MB'].values,axis=0)
table_SEB[2] = np.nanmean(MB_all)

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_output   = 'out'
name_table1  = 'stat_MB'
name_fig     = 'MB_esp'
ds = xr.open_dataset(filename_nc)
MB_all  = np.sum(ds['MB'].values,axis=0)
table_SEB[3] = np.nanmean(MB_all)

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_output   = 'out'
name_table1  = 'stat_MB'
name_fig     = 'MB_esp'
ds = xr.open_dataset(filename_nc)
MB_all  = np.sum(ds['MB'].values,axis=0)
table_SEB[4] = np.nanmean(MB_all)

#filename_nc  = '../../data/output/Peru_output_1_new_20160901-20170831.nc'
filename_nc  = 'in/Peru_output_1_new_20160901-20170831.nc'
dir_output   = 'out'
name_table1  = 'stat_MB'
name_fig     = 'MB_esp'
ds = xr.open_dataset(filename_nc)
MB_all  = np.sum(ds['MB'].values,axis=0)
table_SEB[5] = np.nanmean(MB_all)
pd.DataFrame(table_SEB).to_csv(dir_graphics+"/"+name_f,sep='\t', float_format='%.3f')
