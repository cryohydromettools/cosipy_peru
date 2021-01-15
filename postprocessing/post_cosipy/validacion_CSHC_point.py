import xarray as xr
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

ds = xr.open_dataset('../../data/output/Peru_C0_20160901-20170831.nc')

stakes_loc_file = '../../data/input/Peru/loc_stakes1.csv'
stakes_data_file = '../../data/input/Peru/data_stakes_peru_year1.csv'
df_stakes_loc = pd.read_csv(stakes_loc_file, delimiter='\t', na_values='-9999')
df_stakes_data = pd.read_csv(stakes_data_file, delimiter='\t', index_col='TIMESTAMP', na_values='-9999')
df_stakes_data.index = pd.to_datetime(df_stakes_data.index)
df_stakes_data = df_stakes_data.cumsum(axis=0)
df_stakes_data = df_stakes_data-30

TOTALHEIGHT_mod = ds['TOTALHEIGHT'][:,0,0].to_dataframe()
df = TOTALHEIGHT_mod['TOTALHEIGHT']
df = TOTALHEIGHT_mod-30
df2 = df.loc[df_stakes_data.index]

tot_h_mod = df2['TOTALHEIGHT'].values
tot_h_obs = df_stakes_data['S18'].values

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(df_stakes_data['S18'],'b.',lw=0.4, label='Measured')
ax.plot(df['TOTALHEIGHT'],'r-',lw=0.8, label='Modelled')
ax.plot(df2['TOTALHEIGHT'],'r.',lw=0.4, label='Modelled')
#    ax.set_title(stake_names[i])
ax.set_ylim(-12,0)
ax.set_yticks(np.arange(-12, 2, 2))
ax.set_xlim(pd.Timestamp('2016-09-01'), pd.Timestamp('2017-09-01'))
ax.xaxis.set_tick_params(which='major',rotation=90)
ax.set_ylabel('CSHC (m)')

mod_sta = tot_h_mod
obs_sta = tot_h_obs
corf_pearson = np.corrcoef(obs_sta,mod_sta)
corf_pearson = np.round(corf_pearson[1,0], decimals=3)
rms_stake = np.round(sqrt(mean_squared_error(obs_sta, mod_sta)),decimals=1)
pbias = (np.round(100 *  (np.sum( mod_sta - obs_sta)/np.sum(obs_sta)),decimals=1))


ax.text(pd.Timestamp("2017-07-01"), -2, 'S18')
ax.text(pd.Timestamp("2016-09-10"), -11.5, 'PBIAS ='+' '+str(pbias)+' '+'%')
ax.text(pd.Timestamp("2016-09-10"), -9.8, 'RMSE ='+' '+str(rms_stake)+' '+'m')
ax.text(pd.Timestamp("2016-09-10"), -8.2, str(df_stakes_loc['elev'][17])+' '+'m asl')


font_f = 16
plt.rc('font', size=font_f)          # controls default text sizes
plt.rc('axes', titlesize=font_f)     # fontsize of the axes title
plt.rc('axes', labelsize=font_f)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_f)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_f)    # legend fontsize

fig.savefig('out/val_point.png',dpi = 300, bbox_inches = 'tight', 
             pad_inches = 0.1, format='png')
