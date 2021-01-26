import xarray as xr
import numpy as np

filename_nc  = '../../data/output/Peru_C0_20160901-20170831.nc'

ds = xr.open_dataset(filename_nc)

print(ds.keys())

SWin  = ds['G'].mean()
alpha = ds['ALBEDO'].mean()
LWin  = ds['LWin'].mean()
LWout = ds['LWout'].mean()
Qsens = ds['H'].mean()
Qlat  = ds['LE'].mean()
QR    = ds['B'].mean()
ME    = ds['ME'].mean()

MEc   = SWin*(1-alpha)+LWin+LWout+Qsens+Qlat+QR

print(SWin)
print(alpha)
print(LWin)
print(LWout)
print(Qsens)
print(Qlat)
print(QR)
print(ME)
print(MEc)

T2 = ds['T2'].where(ds['MASK']==1).resample(time='1D').mean('time')
df = T2.mean('lon').mean('lat').to_dataframe()
TS = ds['TS'].where(ds['MASK']==1).resample(time='1D').mean('time')
df['TS'] = TS.mean('lon').mean('lat').values

#df = df-273.16
fig = df.plot(figsize = (10,4)).set_ylabel('Temperature (°K)').get_figure()
fig.savefig('out/temp.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

T2 = ds['T2'].where(ds['MASK']==1)
df = T2.mean('lon').mean('lat').to_dataframe()
TS = ds['TS'].where(ds['MASK']==1)
df['TS'] = TS.mean('lon').mean('lat').values
df = df.loc['2016-12-25':'2017-01-10']
fig = df.plot(figsize = (10,4)).set_ylabel('Temperature (°K)').get_figure()
fig.savefig('out/temp_hour.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)


G = ds['G'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1 = G.to_dataframe()
alpha  = ds['ALBEDO'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['SWnet'] = G * (1 - alpha)
del df1['G']
LWin  = ds['LWin'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
LWout = ds['LWout'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['LWnet'] = LWin+LWout
H  = ds['H'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['Qsens'] = H
LE  = ds['LE'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['Qlat'] = LE
B  = ds['B'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['QG'] = B

QRR = ds['QRR'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['QRR'] = QRR

ME = ds['ME'].where(ds['MASK']==1).resample(time='1D').mean('time').mean('lon').mean('lat')
df1['Qmet'] = ME
fig = df1.plot(figsize = (10,4)).set_ylabel('Energy fluxes (W m$^{-2}$)').get_figure()
fig.savefig('out/seb.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

ALBEDO = ds['ALBEDO'].where(ds['MASK']==1).resample(time='1D').mean('time')
df = ALBEDO.mean('lon').mean('lat').to_dataframe()
fig = df.plot(figsize = (10,4)).set_ylabel('Albedo').get_figure()
fig.savefig('out/albedo.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

TOTAL_P = ds['RRR'].where(ds['MASK']==1).resample(time='1D').sum('time')
df = TOTAL_P.mean('lon').mean('lat').to_dataframe()
RAIN = ds['RAIN'].where(ds['MASK']==1).resample(time='1D').sum('time')
SNOWFALL = ds['SNOWFALL'].where(ds['MASK']==1).resample(time='1D').sum('time')

TOTAL_PT = []
RAIN_T = []
SNOWFALL_T = []
for t in range(len(df)):
    TOTAL_PT.append(np.diagonal(TOTAL_P[t].values).mean())
    RAIN_T.append(np.diagonal(RAIN[t].values).mean())
    SNOWFALL_T.append(np.diagonal(SNOWFALL[t].values).mean()*1000)

df['RRR'] = TOTAL_PT
df['RAIN'] = RAIN_T
df['SNOWFALL'] = SNOWFALL_T

fig = df.plot(figsize = (10,4)).set_ylabel('PP (mm w.e.)').get_figure()
fig.savefig('out/pp_total.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

MB = ds['MB'].where(ds['MASK']==1).resample(time='1D').sum('time')
df = MB.mean('lon').mean('lat').to_dataframe()
surfMB = ds['surfMB'].where(ds['MASK']==1).resample(time='1D').sum('time')
intMB = ds['intMB'].where(ds['MASK']==1).resample(time='1D').sum('time')

MB_T = []
surfMB_T = []
intMB_T = []
for t in range(len(df)):
    MB_T.append(np.diagonal(MB[t].values).mean())
    surfMB_T.append(np.diagonal(surfMB[t].values).mean())
    intMB_T.append(np.diagonal(intMB[t].values).mean())

df['MB'] = MB_T
df['MB_surf'] = surfMB_T
df['MB_sub'] = intMB_T
df = df.rename(columns={'MB': 'MB_total'})

fig = df.plot(figsize = (10,4)).set_ylabel('Mass Balance (m w.e.)').get_figure()
fig.savefig('out/MB_total.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

Melt_surf = ds['surfM'].where(ds['MASK']==1).resample(time='1D').sum('time')
df = Melt_surf.mean('lon').mean('lat').to_dataframe()
subM = ds['subM'].where(ds['MASK']==1).resample(time='1D').sum('time')

surfMB_T = []
intMB_T = []
for t in range(len(df)):
    surfMB_T.append(np.diagonal(Melt_surf[t].values).mean())
    intMB_T.append(np.diagonal(subM[t].values).mean())

df['surfM'] = surfMB_T
df['Melt_sub'] = intMB_T
df = df.rename(columns={'surfM': 'Melt_surf'})

fig = df.plot(figsize = (10,4)).set_ylabel('Melt (m w.e.)').get_figure()
fig.savefig('out/Melt_total.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

print(df.sum())


