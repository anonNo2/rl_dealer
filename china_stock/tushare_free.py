import os

import tushare as ts
import pandas as pd
import numpy as np
import datetime
token = "c23bed479c6639f4ff2ca4a6384fbd3c9595a6c7bb70e815a1b7346d"

pro = ts.pro_api(token)
df = pro.daily(ts_code='000651.SZ', start_date='20060701', end_date='20181218')

filename = "geli000651.SZ.csv"
if os.path.exists(filename):
    df.to_csv(filename, mode='a', header=None)
else:
    df.to_csv(filename)

#df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
#取000001的前复权行情
#df = pro.pro_bar(pro_api=pro, ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')
#df = api.pro_bar(pro_api=api, ts_code='000002.SZ', adj='qfq', start_date='20080101', end_date='20181011')

#df = ts.get_hist_data('300002',ktype ='60')
#df = ts.get_h_data('300002',start='2009-12-16', end='2010-12-16')
