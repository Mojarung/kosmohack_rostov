"""Собирает компактный JSON для интерактивного дашборда из train/test."""
import pandas as pd, numpy as np, json
tr = pd.read_csv('data/train_dataset.csv', parse_dates=['date']); tr['split']='train'; tr['is_synthetic_gap']=False
te = pd.read_csv('data/test_dataset.csv', parse_dates=['date']); te['split']='test'
df = pd.concat([tr, te], ignore_index=True)
df['yr']=df.date.dt.year
df['src']=np.select([df.is_synthetic_gap, df.s2_ndvi.notna(), df.landsat_ndvi.notna(), df.modis_ndvi.notna()], [3,0,1,2], -1)
obs = df[df.src>=0].copy()
obs['z'] = (obs.primary_ndvi-obs.ndvi_climatology_mean)/obs.ndvi_climatology_std

# ERA5 groups (общая ячейка погоды): по идентичности ряда температур на общих датах
dense = df[df.era5_temp_c.notna()]
piv = dense.pivot_table(index='date', columns='anon_polygon_id', values='era5_temp_c')
keys = {}
for c in piv.columns:
    s = piv[c].dropna().round(5)
    # ключ: первые 300 значений 2010 года (все плотные полигоны их имеют, кроме только-2025)
    s10 = s[s.index.year==2010]
    key = tuple(s10.values[:300]) if len(s10) else tuple(s[s.index.year==2025].values[:150])
    keys.setdefault(key, []).append(c)
grp = {}
for i, (k, v) in enumerate(sorted(keys.items(), key=lambda kv: kv[1][0])):
    for c in v: grp[c]=i+1

polys=[]
for pid, d in df.groupby('anon_polygon_id'):
    o = obs[obs.anon_polygon_id==pid]
    tr_years = sorted(d[d.split=='train'].yr.unique().tolist()); te_years = sorted(d[d.split=='test'].yr.unique().tolist())
    polys.append(dict(id=pid, crop=d.crop_type.iloc[0], inTrain=bool(len(tr_years)), years=sorted(d.yr.unique().tolist()),
        dense=bool((d.groupby('yr').size()>=200).any()), nObs=int((o.src<3).sum()), nGaps=int((o.src==3).sum()),
        meanZ=None if o.z.isna().all() else round(float(o.z.mean()),3), shareNeg=None if o.z.isna().all() else round(float((o.z<-1).mean()),3),
        wx=grp.get(pid), split=('both' if tr_years and te_years else ('train' if tr_years else 'test'))))

series={}
for pid, o in obs.groupby('anon_polygon_id'):
    o=o.sort_values('date')
    series[pid]=dict(d=o.date.dt.strftime('%Y-%m-%d').tolist(), v=[None if pd.isna(x) else round(float(x),3) for x in o.primary_ndvi],
        s=o.src.astype(int).tolist(), m=[None if pd.isna(x) else round(float(x),3) for x in o.ndvi_climatology_mean],
        sd=[None if pd.isna(x) else round(float(x),3) for x in o.ndvi_climatology_std])

# погода: по полигону и году, недельные суммы осадков и средние температуры (31 неделя от 1 апреля)
wx={}
w = df[df.era5_temp_c.notna()].copy(); w['wk']=((w.date.dt.dayofyear-91)//7).clip(0,30)
for (pid, yr), d in w.groupby(['anon_polygon_id','yr']):
    g = d.groupby('wk').agg(p=('era5_precip_mm', lambda s: s.clip(lower=0).sum()), t=('era5_temp_c','mean'))
    g = g.reindex(range(31))
    wx.setdefault(pid, {})[str(yr)] = dict(p=[None if pd.isna(x) else round(float(x),1) for x in g.p], t=[None if pd.isna(x) else round(float(x),1) for x in g.t])

# сводка по годам (только train: там есть статус и полная погода)
o = obs[(obs.split=='train')]
ww = df[(df.split=='train')&df.era5_temp_c.notna()].groupby(['anon_polygon_id','yr']).agg(t=('era5_temp_c','mean'), p=('era5_precip_mm', lambda s: s.clip(lower=0).sum())).groupby('yr').mean()
years=[]
for yr, d in o.groupby('yr'):
    years.append(dict(yr=int(yr), meanZ=round(float(d.z.mean()),3), shareNeg=round(float((d.z<-1).mean()),3), shareCrit=round(float((d.z<-2).mean()),3),
        meanNdvi=round(float(d.primary_ndvi.mean()),3), precip=round(float(ww.p.get(yr,np.nan)),0), temp=round(float(ww.t.get(yr,np.nan)),1), n=int(len(d))))
# тепловая карта: полигон × год, средний z (train+test известные)
heat={}
for (pid, yr), d in obs.groupby(['anon_polygon_id','yr']):
    if d.z.notna().any(): heat.setdefault(pid,{})[str(yr)]=round(float(d.z.mean()),2)

# профиль сезона по культурам (бины 8 дней)
o2 = obs[obs.src<3].copy(); o2['b']=(o2.date.dt.dayofyear//8)*8
prof = {c: [[int(b), round(float(g['mean']),3), round(float(g['std']),3)] for b, g in d.groupby('b').primary_ndvi.agg(['mean','std']).iterrows()] for c, d in o2.groupby('crop_type')}
# смещения сенсоров
b1 = tr[tr.s2_ndvi.notna()&tr.landsat_ndvi.notna()]; b2 = tr[tr.s2_ndvi.notna()&tr.modis_ndvi.notna()]
pairs = dict(ls=[[round(float(a),3), round(float(b),3)] for a,b in zip(b1.s2_ndvi, b1.landsat_ndvi)], mo=[[round(float(a),3), round(float(b),3)] for a,b in zip(b2.s2_ndvi, b2.modis_ndvi)])
out = dict(polys=polys, series=series, wx=wx, years=years, heat=heat, prof=prof, pairs=pairs)
json.dump(out, open('eda/dashboard_data.json','w'), ensure_ascii=False, separators=(',',':'))
import os; print('polys', len(polys), 'groups', len(set(grp.values())), 'size MB', round(os.path.getsize('eda/dashboard_data.json')/1e6,2))
print(pd.Series([p['split'] for p in polys]).value_counts().to_dict())
