# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:19:47 2019

@author: torres
"""

import pandas as pd
import numpy as np

def create_date(time_vec):
    
    str_date = pd.to_datetime(time_vec[0])
    end_date = pd.to_datetime(time_vec[-1])
    if str_date.month <= 9:
        str_date_s = str(str_date.year)+'-0'+str(str_date.month)
    else:
        if str_date.month == 12:
            str_date_s = str(str_date.year+1)+'-'+'01'    
        else:
            str_date_s = str(str_date.year)+'-'+str_date.month
    if end_date.month <= 9:
        end_date_s = str(end_date.year)+'-0'+str(end_date.month+1)
    else:
        if end_date.month == 12:
            end_date_s = str(end_date.year+1)+'-'+'01'
        else:
            end_date_s = str(end_date.year)+'-'+str(end_date.month+1)

    date_cre = np.arange(str_date_s, end_date_s, dtype='datetime64[D]')
    
    return date_cre 
    