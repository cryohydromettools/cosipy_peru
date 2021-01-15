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
from scipy.signal import savgol_filter

filename_nc  = '../../data/output/Peru_C0_20160901-20170831.nc'

DATA  = xr.open_dataset(filename_nc)
time_nc = DATA['time'].values

stakes_loc_file = '../../data/input/Peru/loc_stakes1.csv'
stakes_data_file = '../../data/input/Peru/data_stakes_peru_year1.csv'
df_stakes_loc = pd.read_csv(stakes_loc_file, delimiter='\t', na_values='-9999')
df_stakes_data = pd.read_csv(stakes_data_file, delimiter='\t', index_col='TIMESTAMP', na_values='-9999')
df_stakes_data.index = pd.to_datetime(df_stakes_data.index)

df_stakes_data = df_stakes_data.cumsum(axis=0)
df_stakes_data = df_stakes_data-30

TOTALHEIGHT_mod = np.zeros((len(DATA.time), len(df_stakes_loc)))
ix = 0
jy = 0
for i in range(len(df_stakes_loc)):
    TOTALHEIGHT_mod[:,i] = DATA['TOTALHEIGHT'][:,ix,jy].values
    ix = ix+1
    jy = jy+1

df = pd.DataFrame(TOTALHEIGHT_mod,time_nc, columns = df_stakes_loc['id'])

df = df - 30
df2 = df.loc[df_stakes_data.index]

tot_h_mod = df2.values
tot_h_obs = df_stakes_data.values

tot_h_mod1 = tot_h_mod[-1,:]
tot_h_obs1 = tot_h_obs[-1,:]

stake_names = df_stakes_loc['id'].values

r_stake = []
R_stake = []
RMSE_stake = []
PBIAS_stake = []
i = 0
fig, axes = plt.subplots(nrows=6, ncols=3,figsize=(7.0,8.5))
for ax in axes.flat:
    ax.plot(df_stakes_data[stake_names[i]],'b.',lw=0.4, label='Measured')
    ax.plot(df[stake_names[i]],'r-',lw=0.8, label='Modelled')
    ax.plot(df2[stake_names[i]],'r.',lw=0.4, label='Modelled')
#    ax.set_title(stake_names[i])
    ax.set_ylim(-12,0)
    ax.set_yticks(np.arange(-12, 2, 2))
    ax.set_xlim(pd.Timestamp('2016-09-01'), pd.Timestamp('2017-09-01'))
    ax.xaxis.set_tick_params(which='major',rotation=90)
    mod_sta = tot_h_mod[:,i]
    obs_sta = tot_h_obs[:,i]
    corf_pearson = np.corrcoef(obs_sta,mod_sta)
    corf_pearson = np.round(corf_pearson[1,0], decimals=3)
    rms_stake = np.round(sqrt(mean_squared_error(obs_sta, mod_sta)),decimals=1)
    pbias = (np.round(100 *  (np.sum( mod_sta - obs_sta)/np.sum(obs_sta)),decimals=1))
    r_stake.append(corf_pearson)
    R_stake.append(np.round(np.square(corf_pearson),decimals=3))
    RMSE_stake.append(rms_stake)
    PBIAS_stake.append(pbias)
    
    ax.text(pd.Timestamp("2017-07-01"), -2, stake_names[i])
    ax.text(pd.Timestamp("2016-09-10"), -11.5, 'PBIAS ='+' '+str(pbias)+' '+'%')
    ax.text(pd.Timestamp("2016-09-10"), -9.8, 'RMSE ='+' '+str(rms_stake)+' '+'m')
    ax.text(pd.Timestamp("2016-09-10"), -8.2, str(df_stakes_loc['elev'][i])+' '+'m asl')

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

fig.savefig('out/validation_model.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1, format='png')

