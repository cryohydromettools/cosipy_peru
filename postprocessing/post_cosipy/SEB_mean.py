import xarray as xr

filename_nc  = '../../data/output/Peru_C0_20160901-20170831.nc'

ds = xr.open_dataset(filename_nc)

#print(ds.keys())

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

