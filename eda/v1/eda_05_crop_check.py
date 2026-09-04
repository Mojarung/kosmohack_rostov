import pandas as pd, numpy as np
tr = pd.read_csv('data/train_dataset.csv', parse_dates=['date'])
o = tr[tr.primary_ndvi.notna()].copy(); o['yr']=o.date.dt.year; o['doy']=o.date.dt.dayofyear
# per polygon-year: doy of max NDVI (using 16-day bin means to reduce noise)
o['b']=(o.doy//16)*16
g = o.groupby(['crop_type','anon_polygon_id','yr','b']).primary_ndvi.mean().reset_index()
pk = g.loc[g.groupby(['crop_type','anon_polygon_id','yr']).primary_ndvi.idxmax()]
print('doy of seasonal peak (16-day bins) by crop: distribution')
print(pk.groupby('crop_type').b.describe().round(0))
print('\nshare of polygon-years with peak after doy 180 (July+):'); print(pk.assign(late=pk.b>=180).groupby('crop_type').late.mean().round(3))
print('\nsunflower polygons: peak bin by year'); print(pk[pk.crop_type=='подсолнечник'].pivot(index='yr', columns='anon_polygon_id', values='b'))
