import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
C = {'озимая пшеница':'#2a78d6','зерновые':'#eb6834','подсолнечник':'#1baf7a','пастбища/зерновые':'#eda100'}
SC = {'s2':'#2a78d6','landsat':'#eb6834','modis':'#1baf7a'}
plt.rcParams.update({'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','axes.edgecolor':'#c3c2b7','axes.spines.top':False,'axes.spines.right':False,'grid.color':'#e8e7e2','axes.grid':True,'font.size':10})
tr = pd.read_csv('data/train_dataset.csv', parse_dates=['date'])
te = pd.read_csv('data/test_dataset.csv', parse_dates=['date'])
for df in (tr,te):
    df['src'] = np.select([df.s2_ndvi.notna(), df.landsat_ndvi.notna(), df.modis_ndvi.notna()], ['s2','landsat','modis'], 'none')
    df['doy_'] = df.date.dt.dayofyear; df['yr'] = df.date.dt.year

print('##### 1. Sensor-aware baseline on train self-gaps')
rng = np.random.default_rng(0)
tr2 = tr.copy(); tr2['gap']=False
obs_idx = tr2.index[tr2.primary_ndvi.notna()]
hide = rng.choice(obs_idx, size=int(0.15*len(obs_idx)), replace=False)
truth = tr2.loc[hide, ['anon_polygon_id','date','src','primary_ndvi']].copy()
tr2.loc[hide, ['s2_ndvi','landsat_ndvi','modis_ndvi','primary_ndvi']] = np.nan; tr2.loc[hide,'src']='none'; tr2.loc[hide,'gap']=True
# sensor offsets relative to S2, estimated on remaining same-date pairs
vis = tr2[~tr2.gap]
off = {'s2':0.0}
b = vis[vis.s2_ndvi.notna()&vis.landsat_ndvi.notna()]; off['landsat']=(b.landsat_ndvi-b.s2_ndvi).mean()
b = vis[vis.s2_ndvi.notna()&vis.modis_ndvi.notna()]; off['modis']=(b.modis_ndvi-b.s2_ndvi).mean()
print('offsets vs S2:', {k:round(v,4) for k,v in off.items()})

def infer_src(d, r):
    doy = r.date.dayofyear
    ls_dates = d[d.landsat_ndvi.notna()].date; s2_dates = d[d.s2_ndvi.notna()].date
    on_modis = doy % 16 == 1
    on_ls = len(ls_dates)>0 and (((r.date - ls_dates).dt.days % 8)==0).any()
    on_s2 = len(s2_dates)>0 and (((r.date - s2_dates).dt.days % 5)==0).any()
    if on_modis and not on_ls: return 'modis'
    if on_s2 and not on_ls and not on_modis: return 's2'
    if on_ls and not on_s2: return 'landsat'
    if on_ls and on_s2: return 'landsat'  # ambiguous
    if on_modis: return 'modis'
    return 'landsat'

rows=[]
for pid, d in tr2.groupby('anon_polygon_id'):
    d = d.sort_values('date')
    known = d[d.primary_ndvi.notna()]
    kd = known.date.values; kv = known.primary_ndvi.values; ks = known.src.values
    for _, r in d[d.gap].iterrows():
        i = np.searchsorted(kd, np.datetime64(r.date))
        vp, sp, dp = (kv[i-1], ks[i-1], (r.date-pd.Timestamp(kd[i-1])).days) if i>0 else (np.nan,'s2',np.nan)
        vn, sn, dn = (kv[i], ks[i], (pd.Timestamp(kd[i])-r.date).days) if i<len(kd) else (np.nan,'s2',np.nan)
        hs = infer_src(d[d.yr==r.yr], r)
        rows.append((pid, r.date, vp, vn, dp, dn, sp, sn, hs))
nb = pd.DataFrame(rows, columns=['anon_polygon_id','date','vp','vn','dp','dn','sp','sn','hs']).merge(truth, on=['anon_polygon_id','date'])
def rmse(a,b): m=a.notna()&b.notna(); return np.sqrt(((a[m]-b[m])**2).mean())
nb['mean2']=nb[['vp','vn']].mean(axis=1)
# sensor-aware: bring neighbors to S2 scale, average, bring to inferred hidden sensor scale
vp_s2 = nb.vp - nb.sp.map(off); vn_s2 = nb.vn - nb.sn.map(off)
w = nb.dn/(nb.dp+nb.dn)
lin_s2 = np.where(nb.vp.notna()&nb.vn.notna(), vp_s2*w+vn_s2*(1-w), pd.concat([vp_s2,vn_s2],axis=1).mean(axis=1))
nb['aware'] = lin_s2 + nb.hs.map(off)
nb['aware_oracle'] = lin_s2 + nb.src.map(off)
print('hidden sensor inference accuracy:', (nb.hs==nb.src).mean().round(3)); print(pd.crosstab(nb.src, nb.hs))
print('RMSE mean2=%.4f | sensor-aware (inferred)=%.4f | sensor-aware (oracle src)=%.4f' % (rmse(nb.mean2,nb.primary_ndvi), rmse(nb.aware,nb.primary_ndvi), rmse(nb.aware_oracle,nb.primary_ndvi)))
# same-sensor neighbors only variant: nearest known value of the inferred sensor
print('   by true src:'); print(nb.groupby('src').apply(lambda d: pd.Series({'n':len(d),'mean2':rmse(d.mean2,d.primary_ndvi),'aware':rmse(d.aware,d.primary_ndvi)})).round(4))
# also: median-of-window (robust) baseline
print('   share of |err|>0.2 (mean2):', ((nb.mean2-nb.primary_ndvi).abs()>0.2).mean().round(3), ' contribution of those to MSE:', (((nb.mean2-nb.primary_ndvi)**2)[(nb.mean2-nb.primary_ndvi).abs()>0.2].sum()/((nb.mean2-nb.primary_ndvi)**2).sum()).round(3))

print('\n##### 2. Seasonal profiles by crop (train, all sensors on S2 scale? no - raw primary)')
prof = tr[tr.primary_ndvi.notna()].assign(bin=lambda d: (d.doy_//8)*8).groupby(['crop_type','bin']).primary_ndvi.agg(['mean','median','std','count'])
peak = prof.reset_index().loc[lambda d: d.groupby('crop_type')['mean'].idxmax()]
print('peak of mean NDVI by crop:'); print(peak[['crop_type','bin','mean']])
print('min of mean NDVI (Jun-Sep) by crop:'); pp=prof.reset_index(); pp=pp[(pp.bin>=160)&(pp.bin<=260)]; print(pp.loc[pp.groupby('crop_type')['mean'].idxmin()][['crop_type','bin','mean']])

print('\n##### 3. Anomaly status by year / crop / doy (train)')
st = tr[tr.status.notna()]
print(pd.crosstab(st.yr, st.status, normalize='index').round(3))
print(pd.crosstab(st.crop_type, st.status, normalize='index').round(3))
print('by month:'); print(pd.crosstab(st.date.dt.month, st.status, normalize='index').round(3))
print('by sensor:'); print(pd.crosstab(st.src, st.status, normalize='index').round(3))
print('mean zscore by sensor:', st.groupby('src').ndvi_zscore.mean().round(3).to_dict())
# persistence: runs of consecutive negative status per polygon-year
neg = st.assign(neg=st.ndvi_zscore< -1).sort_values(['anon_polygon_id','date'])
runs = neg.groupby(['anon_polygon_id','yr']).neg.apply(lambda s: (s.groupby((s!=s.shift()).cumsum()).transform('size')[s]).max() if s.any() else 0)
print('max run length of consecutive z<-1 obs per polygon-year:', runs.value_counts().sort_index().head(12).to_dict())
print('polygon-years with >=5 consecutive negative obs:', (runs>=5).sum(), 'of', len(runs))
print('  by year:', runs[runs>=5].reset_index().yr.value_counts().sort_index().to_dict())
# isolated single negative observations (likely cloud noise)
iso = neg.groupby(['anon_polygon_id','yr']).neg.apply(lambda s: ((s) & ~s.shift(1,fill_value=False) & ~s.shift(-1,fill_value=False)).sum()).sum()
print('isolated single negative obs (neighbors ok):', iso, 'of', neg.neg.sum(), 'negative obs')

print('\n##### 4. Weather by year (dense polygons, train), Apr-Oct')
wx = tr[tr.era5_temp_c.notna()].groupby(['anon_polygon_id','yr']).agg(t=('era5_temp_c','mean'), p=('era5_precip_mm','sum')).groupby('yr').mean().round(1)
wx['mean_z'] = st.groupby('yr').ndvi_zscore.mean().round(3); wx['share_neg']=(st.ndvi_zscore<-1).groupby(st.yr).mean().round(3)
wx['mean_ndvi']=st.groupby('yr').primary_ndvi.mean().round(3)
print(wx)
print('corr precip vs mean_z:', wx.p.corr(wx.mean_z).round(3), ' temp vs mean_z:', wx.t.corr(wx.mean_z).round(3))
# Growing-season windows: May-Jun precipitation
mj = tr[tr.era5_temp_c.notna()&tr.date.dt.month.isin([4,5,6])].groupby(['anon_polygon_id','yr']).era5_precip_mm.sum().groupby('yr').mean().round(1)
zmj = st[st.date.dt.month.isin([5,6])].groupby('yr').ndvi_zscore.mean().round(3)
print('Apr-Jun precip vs May-Jun mean z:'); print(pd.DataFrame({'p_AMJ':mj,'z_MJ':zmj}))
print('corr:', mj.corr(zmj).round(3))
# precipitation daily distribution: is it daily sum in mm? 
print('precip: share of days <0.1mm:', (tr.era5_precip_mm<0.1).mean().round(3), ' max:', tr.era5_precip_mm.max().round(1), ' negative values:', (tr.era5_precip_mm<0).sum())

print('\n##### 5. climatology variants (bins)')
sub = tr[tr.primary_ndvi.notna()]
def try_bin(fn, label):
    d = sub.assign(b=fn(sub.doy)); g = d.groupby(['anon_polygon_id','b']).primary_ndvi.agg(['mean','std','count'])
    j = d.join(g, on=['anon_polygon_id','b'])
    print(f'{label}: mean abs diff={np.abs(j["mean"]-j.ndvi_climatology_mean).mean():.4f} exact={np.isclose(j["mean"],j.ndvi_climatology_mean,atol=1e-6).mean():.3f}  std diff={np.abs(j["std"]-j.ndvi_climatology_std).mean():.4f}')
try_bin(lambda d: d//8, 'doy//8'); try_bin(lambda d: d//16, 'doy//16'); try_bin(lambda d: (d-1)//16, '(doy-1)//16'); try_bin(lambda d: d//7, 'doy//7'); try_bin(lambda d: d//15, 'doy//15'); try_bin(lambda d: d//10, 'doy//10'); try_bin(lambda d: (d-91)//15, '(doy-91)//15')
try_bin(lambda d: (d-91)//16, '(doy-91)//16'); try_bin(lambda d: (d-91)//14, '(doy-91)//14'); try_bin(lambda d: sub.date.dt.month, 'month')
# stacked sensors ±7
stk = pd.concat([tr[['anon_polygon_id','doy','year',c]].rename(columns={c:'v'}).dropna() for c in ['s2_ndvi','landsat_ndvi','modis_ndvi']])
d = sub[sub.anon_polygon_id=='AOI-0003']; s=stk[stk.anon_polygon_id=='AOI-0003']
res=[]
for _, r in d.iterrows():
    m = (s.doy-r.doy).abs()<=7; res.append((s.loc[m,'v'].mean(), r.ndvi_climatology_mean))
res=np.array(res); print('stacked sensors ±7 (AOI-0003): mean abs diff', np.abs(res[:,0]-res[:,1]).mean().round(4))
# how much does climatology help predicting gaps? corr of clim_mean with primary
print('corr(clim_mean, primary) train:', sub.ndvi_climatology_mean.corr(sub.primary_ndvi).round(3), ' RMSE(primary - clim_mean):', np.sqrt(((sub.primary_ndvi-sub.ndvi_climatology_mean)**2).mean()).round(4))

print('\n##### 6. Test known rows: outliers, 2025 sensor mix, new polygons climatology presence')
print('test known primary <0:', (te.primary_ndvi<0).sum(), ' >1:', (te.primary_ndvi>1).sum())
print(te[te.primary_ndvi>1][['anon_polygon_id','date','landsat_ndvi','primary_ndvi']])
print('test src share by year:'); print(pd.crosstab(te.yr, te.src, normalize='index').round(3).tail(4))
print('test 2025 first/last observation date:', te[(te.yr==2025)&te.primary_ndvi.notna()].date.min().date(), te[(te.yr==2025)&te.primary_ndvi.notna()].date.max().date())
print('test: new polygons with history (rows>1000):', te.groupby('anon_polygon_id').size().gt(1000).sum())

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(9,4.5))
for crop, g in prof.reset_index().groupby('crop_type'):
    ax.plot(g.bin, g['mean'], color=C[crop], lw=2, label=crop)
    ax.fill_between(g.bin, g['mean']-g['std'], g['mean']+g['std'], color=C[crop], alpha=.08)
ax.set_xlabel('день года (doy)'); ax.set_ylabel('primary_ndvi'); ax.set_title('Сезонный профиль NDVI по культурам (train, среднее ± std, бины 8 дней)')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig('eda/v1/figures/seasonal_profile_by_crop.png', dpi=130); plt.close(fig)

fig, axes = plt.subplots(1,2, figsize=(10,4))
b = tr[tr.s2_ndvi.notna()&tr.landsat_ndvi.notna()]
axes[0].scatter(b.s2_ndvi, b.landsat_ndvi, s=6, alpha=.35, color=SC['landsat'], edgecolor='none'); axes[0].plot([0,1],[0,1], color='#52514e', lw=1, ls='--')
axes[0].set_xlabel('s2_ndvi'); axes[0].set_ylabel('landsat_ndvi'); axes[0].set_title(f'Landsat vs S2 в один день (n={len(b)}, смещение +{(b.landsat_ndvi-b.s2_ndvi).mean():.3f})'); axes[0].set_xlim(-.1,1); axes[0].set_ylim(-.1,1)
b = tr[tr.s2_ndvi.notna()&tr.modis_ndvi.notna()]
axes[1].scatter(b.s2_ndvi, b.modis_ndvi, s=6, alpha=.35, color=SC['modis'], edgecolor='none'); axes[1].plot([0,1],[0,1], color='#52514e', lw=1, ls='--')
axes[1].set_xlabel('s2_ndvi'); axes[1].set_ylabel('modis_ndvi'); axes[1].set_title(f'MODIS vs S2 в один день (n={len(b)}, смещение +{(b.modis_ndvi-b.s2_ndvi).mean():.3f})'); axes[1].set_xlim(-.1,1); axes[1].set_ylim(-.1,1)
fig.tight_layout(); fig.savefig('eda/v1/figures/sensor_bias.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9,4.2))
d = tr[(tr.anon_polygon_id=='AOI-0003')&(tr.yr==2021)]
for s in ['s2','landsat','modis']:
    dd = d[d.src==s]; ax.scatter(dd.date, dd.primary_ndvi, s=22, color=SC[s], label=f'{s} (n={len(dd)})', zorder=3)
kd = d[d.primary_ndvi.notna()]
ax.plot(kd.date, kd.ndvi_climatology_mean, color='#52514e', lw=1.5, ls='--', label='климатическая норма')
ax.fill_between(kd.date, kd.ndvi_climatology_mean-kd.ndvi_climatology_std, kd.ndvi_climatology_mean+kd.ndvi_climatology_std, color='#52514e', alpha=.08)
ax.set_title('AOI-0003 (озимая пшеница), сезон 2021: наблюдения по сенсорам и норма'); ax.set_ylabel('primary_ndvi'); ax.legend(frameon=False, ncol=2)
fig.tight_layout(); fig.savefig('eda/v1/figures/example_polygon_2021.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9,3.8))
sh = pd.crosstab(st.yr, st.status, normalize='index')
ax.bar(sh.index, sh['Угнетение биомассы'], color='#eda100', label='Угнетение биомассы (−2 ≤ z < −1)', width=.7)
ax.bar(sh.index, sh['Критическая аномалия'], bottom=sh['Угнетение биомассы'], color='#e34948', label='Критическая аномалия (z < −2)', width=.7)
ax.set_ylabel('доля наблюдений'); ax.set_title('Доля наблюдений с отрицательной аномалией по годам (train)'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('eda/v1/figures/anomaly_share_by_year.png', dpi=130); plt.close(fig)

fig, axes = plt.subplots(1,2, figsize=(10,3.8))
mind = nb[['dp','dn']].min(axis=1).clip(upper=20)
axes[0].hist(mind, bins=np.arange(0.5,21.5,1), color='#2a78d6'); axes[0].set_xlabel('дней до ближайшего известного наблюдения'); axes[0].set_title('Расстояние до соседа (свои гэпы в train)')
err = (nb.mean2-nb.primary_ndvi)
axes[1].hist(err.clip(-.5,.5), bins=60, color='#eb6834'); axes[1].set_xlabel('ошибка baseline (среднее соседей)'); axes[1].set_title(f'Распределение ошибок, RMSE={rmse(nb.mean2,nb.primary_ndvi):.3f}')
fig.tight_layout(); fig.savefig('eda/v1/figures/baseline_errors.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9,3.8))
gy = te[te.is_synthetic_gap].groupby('yr').size()
ax.bar(gy.index, gy.values, color='#2a78d6', width=.7); ax.set_title('Контрольные точки (is_synthetic_gap) в test по годам'); ax.set_ylabel('строк')
fig.tight_layout(); fig.savefig('eda/v1/figures/test_gaps_by_year.png', dpi=130); plt.close(fig)
print('figures saved')
