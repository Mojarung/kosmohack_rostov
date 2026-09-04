import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
tr = pd.read_csv('data/train_dataset.csv', parse_dates=['date'])
te = pd.read_csv('data/test_dataset.csv', parse_dates=['date'])

print('##### 1. primary_ndvi vs sensors (train)')
m = tr.primary_ndvi.notna()
for c in ['s2_ndvi','landsat_ndvi','modis_ndvi']:
    eq = np.isclose(tr.loc[m,c], tr.loc[m,'primary_ndvi'])
    print(f'{c}: present in primary rows {tr.loc[m,c].notna().sum()}, exactly equal {eq.sum()}')
# priority rule?
def rule(r):
    for c in ['s2_ndvi','landsat_ndvi','modis_ndvi']:
        if pd.notna(r[c]): return r[c]
    return np.nan
pred = tr.loc[m].apply(rule, axis=1)
print('coalesce(s2,landsat,modis) == primary:', np.isclose(pred, tr.loc[m,'primary_ndvi']).mean())
# any primary without any sensor?
print('primary present but all sensors NaN:', (m & tr[['s2_ndvi','landsat_ndvi','modis_ndvi']].isna().all(axis=1)).sum())
print('sensor present but primary NaN:', (~m & tr[['s2_ndvi','landsat_ndvi','modis_ndvi']].notna().any(axis=1)).sum())
# which sensor is source
src = np.select([tr.s2_ndvi.notna(), tr.landsat_ndvi.notna(), tr.modis_ndvi.notna()], ['s2','landsat','modis'], 'none')
tr['src'] = src
print(pd.crosstab(tr.year, tr.src))

print('\n##### 2. sensor combos on same date (train)')
combo = tr[['s2_ndvi','landsat_ndvi','modis_ndvi']].notna().astype(int).astype(str).agg(''.join, axis=1)
print(combo.value_counts())
both = tr[tr.s2_ndvi.notna() & tr.landsat_ndvi.notna()]
print('s2 vs landsat same date: n=%d, mean diff s2-ls=%.4f, corr=%.3f, rmse=%.4f' % (len(both), (both.s2_ndvi-both.landsat_ndvi).mean(), both.s2_ndvi.corr(both.landsat_ndvi), np.sqrt(((both.s2_ndvi-both.landsat_ndvi)**2).mean())))
both = tr[tr.s2_ndvi.notna() & tr.modis_ndvi.notna()]
print('s2 vs modis same date: n=%d, mean diff=%.4f, corr=%.3f, rmse=%.4f' % (len(both), (both.s2_ndvi-both.modis_ndvi).mean(), both.s2_ndvi.corr(both.modis_ndvi), np.sqrt(((both.s2_ndvi-both.modis_ndvi)**2).mean())))
both = tr[tr.landsat_ndvi.notna() & tr.modis_ndvi.notna()]
print('landsat vs modis same date: n=%d, mean diff=%.4f, corr=%.3f, rmse=%.4f' % (len(both), (both.landsat_ndvi-both.modis_ndvi).mean(), both.landsat_ndvi.corr(both.modis_ndvi), np.sqrt(((both.landsat_ndvi-both.modis_ndvi)**2).mean())))

print('\n##### 3. outliers')
print('primary_ndvi <0:', (tr.primary_ndvi<0).sum(), ' >1:', (tr.primary_ndvi>1).sum(), ' <-0.2:', (tr.primary_ndvi<-0.2).sum())
print(tr.loc[(tr.primary_ndvi<-0.2)|(tr.primary_ndvi>1), ['anon_polygon_id','date','s2_ndvi','landsat_ndvi','modis_ndvi','primary_ndvi','ndvi_zscore','status']])
print('s2_evi |x|>10:', (tr.s2_evi.abs()>10).sum(), ' landsat_evi >2:', (tr.landsat_evi>2).sum(), ' landsat_ndwi |x|>1.5:', (tr.landsat_ndwi.abs()>1.5).sum())
print('test primary_ndvi <0:', (te.primary_ndvi<0).sum(), ' >1:', (te.primary_ndvi>1).sum())

print('\n##### 4. climatology / zscore / status')
z = (tr.primary_ndvi - tr.ndvi_climatology_mean)/tr.ndvi_climatology_std
print('zscore formula match:', np.isclose(z[m], tr.ndvi_zscore[m]).mean())
st = pd.cut(tr.ndvi_zscore, [-np.inf,-2,-1,np.inf], labels=['Критическая аномалия','Угнетение биомассы','Штатное развитие'], right=False)
print('status thresholds match:', (st[m].astype(str)==tr.status[m]).mean())
print(pd.crosstab(tr.status, pd.cut(tr.ndvi_zscore,[-np.inf,-2,-1,np.inf], right=False)))
print('n_reference_years:', tr.n_reference_years.value_counts().to_dict(), ' test:', te.n_reference_years.value_counts().to_dict())
print('n_ref_years==0 & primary present:', ((tr.n_reference_years==0)&m).sum(), '  n_ref>0 & primary NaN:', ((tr.n_reference_years>0)&~m).sum())
# is climatology per polygon+doy constant across years? 
g = tr[m].groupby(['anon_polygon_id','doy']).ndvi_climatology_mean.nunique()
print('clim_mean unique per (poly,doy): ', g.value_counts().head().to_dict())
g = tr[m].groupby(['anon_polygon_id','year']).ndvi_climatology_mean.nunique()
print('clim_mean unique per (poly,year): describe', g.describe().to_dict())
# Try: climatology = mean of primary over polygon+doy across all years (with window)?
sub = tr[m]
for w in [0, 7, 15, 30]:
    # rolling doy window mean per polygon across years
    res=[]
    for pid, d in sub.groupby('anon_polygon_id'):
        d = d.sort_values('date')
        for i, r in d.head(60).iterrows():
            mask = (d.doy - r.doy).abs()<=w
            res.append((d.loc[mask,'primary_ndvi'].mean(), r.ndvi_climatology_mean))
    res=np.array(res); print(f'window ±{w}: corr with clim_mean {np.corrcoef(res[:,0],res[:,1])[0,1]:.4f}, mean abs diff {np.abs(res[:,0]-res[:,1]).mean():.4f}')

print('\n##### 5. ERA5 shared across polygons?')
p = tr.pivot_table(index='date', columns='anon_polygon_id', values='era5_temp_c')
print('era5_temp unique values per date (describe):', p.nunique(axis=1).describe().to_dict())
print('era5 NaN dates train:', tr[tr.era5_temp_c.isna()].date.dt.year.value_counts().sort_index().to_dict())
print('era5 NaN dates test:', te[te.era5_temp_c.isna()].date.dt.year.value_counts().sort_index().to_dict())
print('era5 NaN by polygon (train) top:', tr[tr.era5_temp_c.isna()].anon_polygon_id.value_counts().head(10).to_dict())

print('\n##### 6. Train polygon coverage')
cov = tr.groupby('anon_polygon_id').agg(rows=('date','size'), ymin=('year','min'), ymax=('year','max'), nyears=('year','nunique'), n_obs=('primary_ndvi','count'), crop=('crop_type','first'), ncrop=('crop_type','nunique'))
cov['obs_per_season']=cov.n_obs/cov.nyears
print(cov)

print('\n##### 7. Test structure')
te['gap']=te.is_synthetic_gap
tcov = te.groupby('anon_polygon_id').agg(rows=('date','size'), ymin=('date', lambda s: s.dt.year.min()), ymax=('date', lambda s: s.dt.year.max()), nyears=('date', lambda s: s.dt.year.nunique()), n_obs=('primary_ndvi','count'), n_gap=('gap','sum'), crop=('crop_type','first'))
tcov['in_train']=tcov.index.isin(tr.anon_polygon_id.unique())
print(tcov)
print(tcov.groupby('in_train')[['rows','n_obs','n_gap']].sum())
print('gap rows by year:', te[te.gap].date.dt.year.value_counts().sort_index().to_dict())
print('all rows by year (test):', te.date.dt.year.value_counts().sort_index().to_dict())
print('gap masked columns NaN share:'); print(te[te.gap].isna().mean().round(3))
print('gap rows crop:', te[te.gap].crop_type.value_counts().to_dict())
print('gap rows doy dist:', te[te.gap].date.dt.dayofyear.describe().to_dict())
