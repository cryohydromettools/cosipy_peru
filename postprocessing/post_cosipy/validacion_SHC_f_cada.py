import pandas as pd
import xarray as xr
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from scipy import spatial
from datetime import datetime
import collections
from math import sqrt
from numpy.polynomial.polynomial import polyfit
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
from scipy.signal import savgol_filter



#filename_nc  = 'output/Peru_out_350m_20160901-20170831.nc'

filename_nc  = '../../data/output/Peru_out_50m_1_20160901-20170831.nc'
dir_output   = 'out'#''
name_table0  = 'stat_MB'
name_figMB   = 'MBtotal'
name_figMB2  = 'MBpos'
name_fig3    = 'SCHS_end'
name_table0  = 'stat_MB'
name_table1  = 'rmse_espacial_mod'
name_table2  = 'rmse_temporal_mod'
name_table3  = 'stat_temporal_stake'
name_table4  = 'stat_espacial_stake'
name_fig     = 'vali_temporal_mod2'
name_fig1    = 'vali_espacial_mod2'

DATA  = xr.open_dataset(filename_nc)
time_nc = DATA['time'].values

#stakes_loc_file = 'output/loc_stakes1.csv'
#stakes_data_file = 'output/data_stakes_peru_year1.csv'

stakes_loc_file = '../../data/input/Peru/loc_stakes1.csv'
stakes_data_file = '../../data/input/Peru/data_stakes_peru_year1.csv'
df_stakes_loc = pd.read_csv(stakes_loc_file, delimiter='\t', na_values='-9999')
df_stakes_data = pd.read_csv(stakes_data_file, delimiter='\t', index_col='TIMESTAMP', na_values='-9999')
df_stakes_data.index = pd.to_datetime(df_stakes_data.index)

#df_stakes_data1 = 30+df_stakes_data

#df_stakes_data[0,:] = DATA.Initial_glacier_height

df_stakes_data = df_stakes_data.cumsum(axis=0)
df_stakes_data = df_stakes_data-30
#df_stakes_data1 = 30+df_stakes_data

#df_stakes_data1.to_csv('output/data_stakes_peru0.csv',sep='\t', float_format='%.2f')

df_stat = pd.DataFrame()
df_val = df_stakes_data.copy()

ME_all   = np.sum(DATA['MB'][:,:,:].values,axis=0)


lon_nc  = DATA['lon'].values
lat_nc  = DATA['lat'].values

statMB  =np.array([np.nanmean(ME_all),np.nanstd(ME_all),np.nanmin(ME_all),np.nanmax(ME_all)])
np.savetxt(dir_output +'/'+ name_table0 +'.csv',statMB, delimiter='\t',fmt='%1.3f')

ME_all2 = np.sum(DATA ['MB'][:,:,:].values,axis=0)
ME_all2[ME_all2<0] = np.nan
ME_all2[ME_all2>0] = 1

# ELA

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r


cmap = mpl.cm.jet
cmap_r = reverse_colourmap(cmap)

fig, ax = plt.subplots(figsize=(4,5))
im  = ax.contourf(lon_nc,lat_nc,ME_all,20,vmin=-10, vmax=1, cmap=cmap_r)

ax.set_yticks(np.round(np.linspace(np.min(lat_nc), np.max(lat_nc), 5), decimals=2))
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_tick_params(which='both', rotation=90)
ax.set_xticks(np.round(np.linspace(np.min(lon_nc), np.max(lon_nc), 5), decimals=2))
ax.yaxis.set_tick_params(which='both', rotation=90)
ax.set_ylabel('Lat (°)')
ax.set_xlabel('Lon (°)')
ax.xaxis.set_label_position('top')
lab_cbar = np.arange(-10,1,2)
fig.colorbar(im, label = 'Mass Balance (m w.e.)', format='%1.1f',
             orientation="horizontal", ticks=lab_cbar)
font_f = 12
plt.rc('font', size=font_f)          # controls default text sizes
plt.rc('axes', titlesize=font_f)     # fontsize of the axes title
plt.rc('axes', labelsize=font_f)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_f)    # legend fontsize
fig.savefig(dir_output+'/'+name_figMB+'.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1)

fig, ax = plt.subplots(figsize=(4,5))
im  = ax.contourf(lon_nc,lat_nc,ME_all2,20,vmin=-10, vmax=1, cmap=cmap_r)

ax.set_yticks(np.round(np.linspace(np.min(lat_nc), np.max(lat_nc), 5), decimals=2))
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_tick_params(which='both', rotation=90)
ax.set_xticks(np.round(np.linspace(np.min(lon_nc), np.max(lon_nc), 5), decimals=2))
ax.yaxis.set_tick_params(which='both', rotation=90)
ax.set_ylabel('Lat (°)')
ax.set_xlabel('Lon (°)')
ax.xaxis.set_label_position('top')
lab_cbar = np.arange(-10,1,2)
fig.colorbar(im, label = 'Mass Balance (m w.e.)', format='%1.1f',
             orientation="horizontal", ticks=lab_cbar)
font_f = 12
plt.rc('font', size=font_f)          # controls default text sizes
plt.rc('axes', titlesize=font_f)     # fontsize of the axes title
plt.rc('axes', labelsize=font_f)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_f)    # legend fontsize
fig.savefig(dir_output+'/'+name_figMB2+'.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1)

stat_MB = np.zeros((1,5))
stat_MB[0,0] = np.nanmean(ME_all)

stat_MB[0,1] = np.nanstd(ME_all)
stat_MB[0,2] = np.nanmin(ME_all)
stat_MB[0,3] = np.nanmax(ME_all)
stat_MB[0,4] = np.nanmedian(ME_all)

df_va1 = pd.DataFrame(stat_MB, columns = ['mean','std','min','max','median'])
df_va1.to_csv(dir_output +'/'+ name_table0 +'.csv',sep='\t', float_format='%.2f')


def transform_coordinates(coords):
    """ Transform coordinates from geodetic to cartesian
    an array of tuples)
    """
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]   
    E2 = 6.69437999014e-3 # eccentricity squared    
    
    coords = np.asarray(coords).astype(np.float)
                                                      
    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])
    
    # convert to radiants
    lat_rad = np.radians(coords[:,0])
    lon_rad = np.radians(coords[:,1]) 
    
    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)
    
    return np.column_stack((x, y, z))


lon_mesh, lat_mesh = np.meshgrid(DATA.lon.values, DATA.lat.values)

coords = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

ground_pixel_tree = spatial.cKDTree(transform_coordinates(coords))

 # Check for stake data
stakes_list = []
for index, row in df_stakes_loc.iterrows():
    index = ground_pixel_tree.query(transform_coordinates((row['lat'], row['lon'])))
    index = np.unravel_index(index[1], lat_mesh.shape)
    stakes_list.append((index[0][0], index[1][0], row['id']))


northing = 'lat'	                                    # name of dimension	in in- and -output
easting  = 'lon'					                        # name of dimension in in- and -output
stake_names = []
TOTALHEIGHT_mod = np.zeros((len(DATA.time), len(stakes_list)))
for i in range(len(stakes_list)):
    for j in range(len(DATA.time)):
        tp1 = stakes_list[i]
        stake_loc_y = tp1[0]
        stake_loc_x = tp1[1]
        stake_name  = tp1[2]
        TOTALHEIGHT_mod[j,i] = DATA.TOTALHEIGHT.values[j,stake_loc_y,stake_loc_x]
    stake_names.append(stake_name)


df = pd.DataFrame(TOTALHEIGHT_mod,time_nc, columns = stake_names)

df = df - 30
df2 = df.loc[df_stakes_data.index]

tot_h_mod = df2.values
tot_h_obs = df_stakes_data.values

tot_h_mod1 = tot_h_mod[-1,:]
tot_h_obs1 = tot_h_obs[-1,:]

fig, (ax0) = plt.subplots(figsize=(5,5))    
tot_h_mod_mean_esp = tot_h_mod1
tot_h_obs_mean_esp = tot_h_obs1
corf_pearson = np.corrcoef(tot_h_obs_mean_esp,tot_h_mod_mean_esp)
corf_pearson = corf_pearson[1,0]
b, m = polyfit(tot_h_obs_mean_esp, tot_h_mod_mean_esp, 1)
ax0.plot(tot_h_obs_mean_esp, tot_h_mod_mean_esp, 'k.', markersize=2)
ax0.plot(tot_h_obs_mean_esp, b + m * tot_h_obs_mean_esp, 'k-', lw=0.4)
ax0.set_yticks(np.arange(-14, 4, 2))
ax0.set_ylim((-14,-4))
ax0.set_xticks(np.arange(-14, 4, 2))
ax0.set_xlim((-14,-4))
ax0.text(-13, -6, 'r ='+' '+str(np.round(corf_pearson,decimals=2)))
ax0.text(-13, -6.8, 'R$^{2}$ ='+' '+str(np.round(np.square(corf_pearson),decimals=2)))
ax0.text(-13, -7.6, '$p$ <'+' '+'0.05')
ax0.set_xlabel('CSHC measured (m)')
ax0.set_ylabel('CSHC modelled (m)')
font_f = 12
plt.rc('font', size=font_f)          # controls default text sizes
plt.rc('axes', titlesize=font_f)     # fontsize of the axes title
plt.rc('axes', labelsize=font_f)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_f)    # legend fontsize

fig.savefig(dir_output+'/'+name_fig3+'.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1)


i = 0
fig, axes = plt.subplots(nrows=5, ncols=4,figsize=(8.5,10.5))
for ax in axes.flat:
    ax.plot(df[stake_names[i]],'r',lw=0.4, label='Modelled')
    ax.plot(df_stakes_data[stake_names[i]],'b',lw=0.4, label='Measured')
    ax.set_title(stake_names[i])
    ax.set_ylim(-12,0)
    ax.set_yticks(np.arange(-12, 2, 2))
    ax.set_xlim(pd.Timestamp('2016-09-01'), pd.Timestamp('2017-09-01'))
    ax.xaxis.set_tick_params(which='major',rotation=90)
    mod_sta = tot_h_mod[:,i]
    obs_sta = tot_h_obs[:,i]
    corf_pearson = np.corrcoef(obs_sta,mod_sta)
    corf_pearson = corf_pearson[1,0]
    rms_stake = np.round(sqrt(mean_squared_error(obs_sta, mod_sta)),decimals=1)
    pbias = (np.round(100 *  (np.sum( mod_sta - obs_sta)/np.sum(obs_sta)),decimals=1))
    ax.text(pd.Timestamp("2016-09-10"), -11.5, 'PBIAS ='+' '+str(pbias)+' '+'%')
    ax.text(pd.Timestamp("2016-09-10"), -10, 'RMSE ='+' '+str(rms_stake)+' '+'m')
    ax.text(pd.Timestamp("2016-09-10"), -8.5, str(df_stakes_loc['elev'][i])+' '+'m asl')

    i = i+1
    
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
entries = collections.OrderedDict()
for ax in axes.flatten():
  for handle, label in zip(*ax.get_legend_handles_labels()):
    entries[label] = handle

legend = fig.legend(entries.values(), entries.keys(),
    loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.04))

for ax in axes.flat:
    ax.label_outer()

fig.text(0.05, 0.5, 'Cumulative surface height change (m)', va='center', rotation='vertical')

font_f = 10
plt.rc('font', size=font_f)          # controls default text sizes
plt.rc('axes', titlesize=font_f)     # fontsize of the axes title
plt.rc('axes', labelsize=font_f)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_f)    # legend fontsize

fig.savefig(dir_output+'/'+name_fig+'.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1)

rms_tim, rms_esp   = np.shape(tot_h_mod)

stat_MB_table_esp = np.zeros((rms_esp,4))
for i in range(rms_esp):
    corf_pearson = np.corrcoef(tot_h_obs[:,i],tot_h_mod[:,i])
    corf_pearson = corf_pearson[1,0]
    rms_stake = np.round(sqrt(mean_squared_error(tot_h_obs[:,i], tot_h_mod[:,i])),decimals=2)
    pbias = (np.round(100 *  (np.sum(tot_h_mod[:,i] - tot_h_obs[:,i])/np.sum(tot_h_obs[:,i])),decimals=2))
    stat_MB_table_esp[i,0] = np.round(corf_pearson, decimals=2)
    stat_MB_table_esp[i,1] = np.round(corf_pearson**2, decimals=2)
    stat_MB_table_esp[i,2] = rms_stake
    stat_MB_table_esp[i,3] = pbias

df_va1 = pd.DataFrame(stat_MB_table_esp,stake_names, columns = ['r','R','rmse','pbias'])
df_va1.to_csv(dir_output +'/'+ name_table1 +'.csv',sep='\t', float_format='%.2f')


stat_MB_table_temp = np.zeros((rms_tim,4))

for i in range(rms_tim):
    corf_pearson = np.corrcoef(tot_h_obs[i,:],tot_h_mod[i,:])
    corf_pearson = corf_pearson[1,0]
    rms_stake = np.round(sqrt(mean_squared_error(tot_h_obs[i,:], tot_h_mod[i,:])),decimals=2)
    pbias = (np.round(100 *  (np.sum(tot_h_mod[i,:] - tot_h_obs[i,:])/np.sum(tot_h_obs[i,:])),decimals=2))
    stat_MB_table_temp[i,0] = np.round(corf_pearson, decimals=2)
    stat_MB_table_temp[i,1] = np.round(corf_pearson**2, decimals=2)
    stat_MB_table_temp[i,2] = rms_stake
    stat_MB_table_temp[i,3] = pbias

df_va2 = pd.DataFrame(stat_MB_table_temp,df_stakes_data.index, columns = ['r','R','rmse','pbias'])
df_va2.to_csv(dir_output +'/'+ name_table2 +'.csv',sep='\t', float_format='%.2f')



fig, (ax0, ax1) = plt.subplots(1,2,figsize=(11,5))    

tot_h_mod_mean_esp = np.nanmean(tot_h_mod,axis=1)
tot_h_obs_mean_esp = np.nanmean(tot_h_obs,axis=1)
rms_table_tim = sqrt(mean_squared_error(tot_h_obs[i,:], tot_h_mod[i,:]))
b, m = polyfit(tot_h_obs_mean_esp, tot_h_mod_mean_esp, 1)
ax0.plot(tot_h_obs_mean_esp, tot_h_mod_mean_esp, 'k.', markersize=2)
ax0.plot(tot_h_obs_mean_esp, b + m * tot_h_obs_mean_esp, 'k-', lw=0.4)
ax0.set_yticks(np.arange(-12, 4, 2))
ax0.set_ylim((-10,2))
ax0.set_xticks(np.arange(-12, 4, 2))
ax0.set_xlim((-10,2))
ax0.set_xlabel('CSHC med (m)')
ax0.set_ylabel('CSHC mod (m)')
ax0.text(-9.3, 1.0, '(a)')

tot_h_mod_mean_esp = np.nanmean(tot_h_mod,axis=0)
tot_h_obs_mean_esp = np.nanmean(tot_h_obs,axis=0)
rms_table_tim = sqrt(mean_squared_error(tot_h_obs[i,:], tot_h_mod[i,:]))
b, m = polyfit(tot_h_obs_mean_esp, tot_h_mod_mean_esp, 1)
ax1.plot(tot_h_obs_mean_esp, tot_h_mod_mean_esp, 'k.', markersize=2)
ax1.plot(tot_h_obs_mean_esp, b + m * tot_h_obs_mean_esp, 'k-', lw=0.4)
ax1.set_yticks(np.arange(-12, 4, 2))
ax1.set_ylim((-10,2))
ax1.set_xticks(np.arange(-12, 4, 2))
ax1.set_xlim((-10,2))
ax1.set_xlabel('CSHC med (m)')
ax1.set_ylabel('CSHC mod (m)')
ax1.text(-9.3, 1.0, '(b)')

font_f = 12
plt.rc('font', size=font_f)          # controls default text sizes
plt.rc('axes', titlesize=font_f)     # fontsize of the axes title
plt.rc('axes', labelsize=font_f)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_f)    # legend fontsize

fig.savefig(dir_output+'/'+name_fig1+'.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1)

tot_h_mod_mean_tem = np.nanmean(tot_h_mod,axis=1)
tot_h_obs_mean_tem = np.nanmean(tot_h_obs,axis=1)

corf_pearson = np.corrcoef(tot_h_obs_mean_tem,tot_h_mod_mean_tem)
corf_pearson = corf_pearson[1,0]
rms_table_tim = sqrt(mean_squared_error(tot_h_obs_mean_tem, tot_h_mod_mean_tem))
pbias = 100 *  (np.sum( tot_h_mod_mean_tem - tot_h_obs_mean_tem)/np.sum(tot_h_obs_mean_tem))


stat_MB_tem = np.zeros((1,4))
stat_MB_tem[0,0] = corf_pearson
stat_MB_tem[0,1] = corf_pearson**2
stat_MB_tem[0,2] = rms_table_tim
stat_MB_tem[0,3] = pbias

df_va1 = pd.DataFrame(stat_MB_tem, columns = ['r','R','rmse','pbias'])
df_va1.to_csv(dir_output +'/'+ name_table3 +'.csv',sep='\t', float_format='%.2f')


tot_h_mod_mean_esp = np.nanmean(tot_h_mod,axis=0)
tot_h_obs_mean_esp = np.nanmean(tot_h_obs,axis=0)

corf_pearson = np.corrcoef(tot_h_obs_mean_esp,tot_h_mod_mean_esp)
corf_pearson = corf_pearson[1,0]
rms_table_tim = sqrt(mean_squared_error(tot_h_obs_mean_esp, tot_h_mod_mean_esp))
pbias = 100 *  (np.sum( tot_h_mod_mean_esp - tot_h_obs_mean_esp)/np.sum(tot_h_obs_mean_esp))

stat_MB_esp = np.zeros((1,4))
stat_MB_esp[0,0] = corf_pearson
stat_MB_esp[0,1] = corf_pearson**2
stat_MB_esp[0,2] = rms_table_tim
stat_MB_esp[0,3] = pbias

df_va1 = pd.DataFrame(stat_MB_esp, columns = ['r','R','rmse','pbias'])
df_va1.to_csv(dir_output +'/'+ name_table4 +'.csv',sep='\t', float_format='%.2f')
