#!/bin/bash

export PATH="/nfsdata/programs/anaconda3_201812/bin:$PATH"

python MB_date_altitude.py
python MB_espacial.py
python MB_profile_altitude.py
python MB_table2.py
python MB_temporal_daily.py
python SEB_date_altitude.py
python SEB_espacial.py
python SEB_profile_altitude.py
python SEB_table1.py
python SEB_temporal_daily.py
