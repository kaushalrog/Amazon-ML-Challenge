from pathlib import Path
import time,json,sys
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;B=R/'experiments/astra_baseline';t=time.time()
features=['name_jw','name_tsr','name_qratio','name_jac','name_len_ratio','addr_jw','addr_tsr','addr_jac','addr_len_ratio','num_jac','country_eq','is_s3','addr_missing','num_conflict','name_x_addr']
x=pl.read_parquet(R/'experiments/astra_003/mined_train.parquet');model=lgb.LGBMClassifier(n_estimators=100,learning_rate=.15,num_leaves=31,min_child_samples=80,n_jobs=2,random_state=20260927,verbosity=-1);model.fit(x.select(features).to_numpy(),x['label'].to_numpy(),sample_weight=x['_weight'].to_numpy(),feature_name=features);model.booster_.save_model(str(O/'blocker.txt'));rows=[]
for name,path in [('dev',B/'cache/feat_dev_20000_200.parquet'),('broad_pilot',R/'experiments/astra_009/features.parquet')]:
 x=pl.read_parquet(path);p=model.predict_proba(x.select(features).to_numpy())[:,1];s=x.select('s1','src','eid','label').with_columns(pl.Series('fast_score',p.astype('float32')));s.write_parquet(O/f'{name}_scores.parquet')
 for threshold in [1e-6,1e-5,.0001,.0005,.001,.005,.01]:
  z=s.filter(pl.col('fast_score')>=threshold);rows.append({'scope':name,'threshold':threshold,'before_pairs':s.height,'after_pairs':z.height,'positive_retention':z['label'].sum()/s['label'].sum(),'positive_lost':s['label'].sum()-z['label'].sum()})
pl.DataFrame(rows).write_csv(O/'ablation.csv');(O/'manifest.json').write_text(json.dumps({'features':features,'training':'all positives and hard negatives; weighted easy-negative sample','runtime':time.time()-t},indent=2));print(pl.DataFrame(rows),flush=True)
