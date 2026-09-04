import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
tr = pd.read_csv('data/train_dataset.csv', parse_dates=['date'])
te = pd.read_csv('data/test_dataset.csv', parse_dates=['date'])
tr['src'] = np.select([tr.s2_ndvi.notna(), tr.landsat_ndvi.notna(), tr.modis_ndvi.notna()], ['s2','landsat','modis'], 'none')
te['src'] = np.select([te.s2_ndvi.notna(), te.landsat_ndvi.notna(), te.modis_ndvi.notna()], ['s2','landsat','modis'], 'none')

print('##### A. sparse vs dense polygons and ERA5')
for name, df in [('train',tr),('test',te)]:
    g = df.groupby(['anon_polygon_id', df.date.dt.year]).agg(rows=('date','size'), era5_nan=('era5_temp_c', lambda s: s.isna().mean()), obs=('primary_ndvi','count'))
    g['dense'] = g.rows>=200
    print(name, 'polygon-years dense:', g.dense.sum(), 'sparse:', (~g.dense).sum())
    print(name, 'era5 NaN share | dense:', g[g.dense].era5_nan.mean().round(3), '| sparse:', g[~g.dense].era5_nan.mean().round(3))
    dense_rows = df[df.groupby(['anon_polygon_id', df.date.dt.year]).date.transform('size')>=200]
    print(name, 'era5 NaN within dense polygon-years:', dense_rows.era5_temp_c.isna().sum(), 'of', len(dense_rows))
    if dense_rows.era5_temp_c.isna().sum():
        print(dense_rows[dense_rows.era5_temp_c.isna()].groupby(['anon_polygon_id']).date.agg(['min','max','size']).head(10))
# ERA5 grid groups: cluster polygons by identical temp series
p = tr.pivot_table(index='date', columns='anon_polygon_id', values='era5_temp_c')
p = p.dropna(axis=1, how='all')
groups = {}
for c in p.columns:
    key = tuple(p[c].round(6).fillna(-999).values[:400])
    groups.setdefault(key, []).append(c)
print('ERA5 identical-series groups in train:', len(groups)); print([v for v in groups.values()])
# test 2025 era5
t25 = te[te.date.dt.year==2025]
print('test 2025 era5 NaN by date (first/last date with data):', t25[t25.era5_temp_c.notna()].date.min().date(), t25[t25.era5_temp_c.notna()].date.max().date())
print('test 2025 dense polygons era5 NaN share:', t25[t25.groupby('anon_polygon_id').date.transform('size')>=200].era5_temp_c.isna().mean().round(3))
print('test era5 NaN in non-gap rows, dense polygon-years (excluding 2025):')
d = te[(te.date.dt.year<2025) & ~te.is_synthetic_gap]
d = d[d.groupby(['anon_polygon_id', d.date.dt.year]).date.transform('size')>=200]
print(d.era5_temp_c.isna().sum(), 'of', len(d))

print('\n##### B. sensor cadence')
print('MODIS doy mod 16 (train):', (tr[tr.src=='modis'].doy % 16).value_counts().head(5).to_dict())
print('MODIS doy values:', sorted(tr[tr.src=='modis'].doy.unique())[:20])
# Landsat: day gaps between consecutive landsat obs within polygon-year
ls = tr[tr.landsat_ndvi.notna()].sort_values(['anon_polygon_id','date'])
dd = ls.groupby(['anon_polygon_id','year']).date.diff().dt.days
print('Landsat consecutive-day gaps:', dd.value_counts().head(8).to_dict())
s2 = tr[tr.s2_ndvi.notna()].sort_values(['anon_polygon_id','date'])
dd = s2.groupby(['anon_polygon_id','year']).date.diff().dt.days
print('S2 consecutive-day gaps:', dd.value_counts().head(8).to_dict())
print('S2 first year:', s2.year.min(), ' landsat by year first:', ls.year.min())
print('sensor day-of-week? no. Source share by year(train):'); print(pd.crosstab(tr.year, tr.src, normalize='index').round(3))

print('\n##### C. Gap selection: share of gaps among observation rows, by src of surrounding?')
te['is_obs'] = te.primary_ndvi.notna() | te.is_synthetic_gap
print('overall gap share of obs rows:', te.is_synthetic_gap.sum()/te.is_obs.sum())
g = te[te.is_obs].groupby('anon_polygon_id').is_synthetic_gap.mean()
print('gap share per polygon: ', g.describe().round(3).to_dict())
g = te[te.is_obs].groupby(te.date.dt.year).is_synthetic_gap.mean()
print('gap share per year:', g.round(3).to_dict())
# inferring hidden sensor from cadence: does gap date match landsat schedule (date ≡ known landsat date mod 8 within same polygon-year)?
def infer(df):
    out=[]
    for (pid,yr), d in df.groupby(['anon_polygon_id', df.date.dt.year]):
        gaps = d[d.is_synthetic_gap]
        if gaps.empty: continue
        ls_dates = d[d.landsat_ndvi.notna()].date
        s2_dates = d[d.s2_ndvi.notna()].date
        mo_dates = d[d.modis_ndvi.notna()].date
        for _, r in gaps.iterrows():
            doy = r.date.dayofyear
            on_modis = (doy % 16 == 1)
            on_ls = len(ls_dates) and (((r.date - ls_dates).dt.days % 8)==0).any()
            on_s2 = len(s2_dates) and (((r.date - s2_dates).dt.days % 5)==0).any()
            out.append((pid, yr, r.date, on_modis, bool(on_ls), bool(on_s2)))
    return pd.DataFrame(out, columns=['pid','yr','date','on_modis','on_ls','on_s2'])
inf = infer(te)
print('gap rows on modis schedule:', inf.on_modis.mean().round(3), ' on landsat 8-day grid:', inf.on_ls.mean().round(3), ' on S2 5-day grid:', inf.on_s2.mean().round(3))
print(pd.crosstab([inf.on_modis, inf.on_ls], inf.on_s2))
# sanity: same inference on train known rows (where src known) — how well does it identify src?
tr2 = tr.copy(); tr2['is_synthetic_gap']=False
rng = np.random.default_rng(0)
obs_idx = tr2.index[tr2.primary_ndvi.notna()]
hide = rng.choice(obs_idx, size=int(0.15*len(obs_idx)), replace=False)
truth = tr2.loc[hide, ['src','primary_ndvi']].copy()
tr2.loc[hide, ['s2_ndvi','landsat_ndvi','modis_ndvi','primary_ndvi']] = np.nan
tr2.loc[hide, 'is_synthetic_gap']=True
inf2 = infer(tr2).set_index(['pid','date'])
truth = truth.join(tr2.loc[hide, ['anon_polygon_id','date']]).set_index(['anon_polygon_id','date'])
truth.index.names=['pid','date']
j = truth.join(inf2)
print('train self-gaps: sensor vs schedule flags'); print(pd.crosstab(j.src, [j.on_modis, j.on_ls, j.on_s2]))

print('\n##### D. Baselines on self-made gaps (train) and neighbor distances in test')
def neighbors(df, gapmask):
    # for each gap row: prev/next known primary in same polygon
    res=[]
    for pid, d in df.groupby('anon_polygon_id'):
        d = d.sort_values('date')
        known = d[d.primary_ndvi.notna()]
        kd = known.date.values; kv = known.primary_ndvi.values
        for _, r in d[gapmask.loc[d.index]].iterrows():
            i = np.searchsorted(kd, np.datetime64(r.date))
            prev = (kd[i-1], kv[i-1]) if i>0 else (None,np.nan)
            nxt = (kd[i], kv[i]) if i<len(kd) else (None,np.nan)
            dp = (r.date - pd.Timestamp(prev[0])).days if prev[0] is not None else np.nan
            dn = (pd.Timestamp(nxt[0]) - r.date).days if nxt[0] is not None else np.nan
            res.append((pid, r.date, prev[1], nxt[1], dp, dn))
    return pd.DataFrame(res, columns=['pid','date','vprev','vnext','dprev','dnext'])
nb = neighbors(tr2, tr2.is_synthetic_gap)
nb = nb.set_index(['pid','date']).join(truth)
nb['mean2'] = nb[['vprev','vnext']].mean(axis=1)
w = nb.dnext/(nb.dprev+nb.dnext)
nb['linear'] = np.where(nb.vprev.notna()&nb.vnext.notna(), nb.vprev*w + nb.vnext*(1-w), nb.mean2)
nb['nearest'] = np.where(nb.dprev<=nb.dnext, nb.vprev, nb.vnext); nb['nearest']=nb.nearest.fillna(nb.mean2)
def rmse(a,b): m=a.notna()&b.notna(); return np.sqrt(((a[m]-b[m])**2).mean())
print('train self-gaps (15%%): n=%d  RMSE mean2=%.4f  linear=%.4f  nearest=%.4f' % (len(nb), rmse(nb.mean2,nb.primary_ndvi), rmse(nb.linear,nb.primary_ndvi), rmse(nb.nearest,nb.primary_ndvi)))
# clipped-outlier RMSE
cl = nb.primary_ndvi.clip(-0.1, 1)
print('   same, truth clipped to [-0.1,1]: mean2=%.4f linear=%.4f' % (rmse(nb.mean2, cl), rmse(nb.linear, cl)))
print('   by hidden sensor RMSE (linear):'); print(nb.groupby('src').apply(lambda d: pd.Series({'n':len(d),'rmse_lin':rmse(d.linear,d.primary_ndvi),'bias':(d.primary_ndvi-d.linear).mean()})).round(4))
print('   by year RMSE (linear):'); print(nb.reset_index().assign(y=lambda d: d.date.dt.year).groupby('y').apply(lambda d: rmse(d.linear,d.primary_ndvi)).round(4).to_dict())
print('   neighbor dist (self-gaps): dprev', nb.dprev.describe().round(1).to_dict()); 
print('   min(dprev,dnext) quantiles:', nb[['dprev','dnext']].min(axis=1).quantile([.1,.25,.5,.75,.9,.99]).to_dict())
# error vs distance
nb['mind']=nb[['dprev','dnext']].min(axis=1)
print('   RMSE linear by min neighbor distance bucket:'); print(nb.groupby(pd.cut(nb.mind,[0,1,2,4,8,16,32,400])).apply(lambda d: pd.Series({'n':len(d),'rmse':rmse(d.linear,d.primary_ndvi)})).round(4))
# largest errors
print('   worst 10 self-gaps:'); print(nb.assign(err=(nb.linear-nb.primary_ndvi).abs()).sort_values('err',ascending=False).head(10)[['vprev','vnext','dprev','dnext','primary_ndvi','src','err']])

nbt = neighbors(te, te.is_synthetic_gap)
print('\ntest gaps: n=%d, no prev=%d, no next=%d' % (len(nbt), nbt.vprev.isna().sum(), nbt.vnext.isna().sum()))
print('test min neighbor distance quantiles:', nbt[['dprev','dnext']].min(axis=1).quantile([.1,.25,.5,.75,.9,.99]).to_dict())
print('test dprev+dnext quantiles:', (nbt.dprev+nbt.dnext).quantile([.1,.25,.5,.75,.9,.99]).to_dict())
print('test gaps with both neighbors within 5 days:', ((nbt.dprev<=5)&(nbt.dnext<=5)).mean().round(3))
print('test |vprev-vnext| quantiles:', (nbt.vprev-nbt.vnext).abs().quantile([.5,.75,.9,.95,.99]).round(3).to_dict())
# adjacent gaps (consecutive gaps within polygon)
tg = te[te.is_synthetic_gap].sort_values(['anon_polygon_id','date'])
dg = tg.groupby('anon_polygon_id').date.diff().dt.days
print('test gap-to-gap day diff:', dg.value_counts().head(6).to_dict())

print('\n##### E. climatology formula search')
sub = tr[tr.primary_ndvi.notna()].copy()
def clim_variants(d, w, excl_year, stat):
    out=[]
    d = d.sort_values('date')
    for _, r in d.iterrows():
        mask = (d.doy - r.doy).abs()<=w
        if excl_year: mask &= d.year!=r.year
        v = d.loc[mask,'primary_ndvi']
        out.append((v.mean() if stat=='mean' else v.median(), v.std(ddof=1), v.std(ddof=0), r.ndvi_climatology_mean, r.ndvi_climatology_std, d.loc[mask,'year'].nunique(), r.n_reference_years))
    return np.array(out, dtype=float)
sample = sub[sub.anon_polygon_id.isin(['AOI-0003','AOI-0019','AOI-0037'])]
for w in [5,6,7,8,10]:
    for excl in [False, True]:
        r = np.vstack([clim_variants(d, w, excl, 'mean') for _, d in sample.groupby('anon_polygon_id')])
        print(f'w=±{w} excl_year={excl}: mean abs diff mean={np.nanmean(np.abs(r[:,0]-r[:,3])):.4f} exact={np.isclose(r[:,0],r[:,3],atol=1e-6).mean():.3f} | std1 diff={np.nanmean(np.abs(r[:,1]-r[:,4])):.4f} std0 diff={np.nanmean(np.abs(r[:,2]-r[:,4])):.4f} | nyears match={np.mean(r[:,5]==r[:,6]):.3f}')
