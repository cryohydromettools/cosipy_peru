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
# Aggregation to hourly data
#------------------------
aggregate = True
aggregation_step = 'H'

# Delimiter in csv file
delimiter = '\t'

# WRF non uniform grid
WRF = False

#------------------------
# Radiation module 
#------------------------
radiationModule = True 

# Time zone
timezone_lon = -90.0

# Zenit threshold (>threshold == zenit)
zeni_thld = 85.0

# Albedo timescale (0 to 50)
albedo_timescale_1 = 1.5  # 

# Albedo ice (0.1 to 0.4)
albedo_ice = 0.20

# Inicial snow layer
int_snowheight = 0.1

#------------------------
# Point model 
#------------------------
point_model = False 
plon =  -8.965561
plat = -77.629209
hgt  = 4910.0 

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'Artesanraju'
stationAlt = 4910.0

lapse_T         = -0.0055  # Temp K per  m
lapse_RH        = 0.000    # RH % per  m (0 to 1)
lapse_RRR       = 0.0000   # RRR % per m (0 to 1)
lapse_SNOWFALL  = 0.000    # Snowfall % per m (0 to 1)
lapse_albedo_1  = 0.005    # Albedo days per m (0 to 50)
lapse_snow      = 0.002    # Snow layer m per m
lapse_ice       = 0.0001   # albedo ice (0.0001)
