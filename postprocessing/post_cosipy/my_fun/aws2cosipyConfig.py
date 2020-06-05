"""
 This is the configuration (init) file for the utility cs2posipy.
 Please make your changes here.
"""

#------------------------
# Declare variable names 
#------------------------

# Pressure
PRES_var = 'Press_aws'   

# Temperature
T2_var = 'Tair_aws'  
in_K = True 

# Cloud cover fraction
N_var = 'CCF_aws'

# Relative humidity
RH2_var = 'RH_aws'   

# Incoming shortwave radiation
G_var = 'SWin_aws'   

# Precipitation
RRR_var = 'Ptotal_aws' 

# Wind velocity
U2_var = 'ws_aws'     

# Incoming longwave radiation
LWin_var = 'LWinCor_Avg'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

#------------------------
# Radiation module 
#------------------------
radiationModule = True 

# Time zone
timezone_lon = -90.0

# Zenit threshold (>threshold == zenit)
zeni_thld = 85.0

#------------------------
# Point model 
#------------------------
point_model = False 
plon = 10.7779
plat = 46.807984
hgt = 2970 

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Artesanraju'
stationAlt = 4910.0

lapse_T         = -0.006  # Temp K per  m
lapse_RH        = 0.002  # RH % per  m (0 to 1)
lapse_RRR       = 0.0001   # RRR % per m (0 to 1)
lapse_SNOWFALL  = 0.0001   # Snowfall % per m (0 to 1)
