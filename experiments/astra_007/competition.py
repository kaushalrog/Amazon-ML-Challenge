"""Measure whether a candidate has a materially stronger competing S1 reference.

Uses supplied reference text only, never another S1's match label. This is an
abstention diagnostic, not a one-to-one assignment constraint. Candidate count
is fixed equally for current/alternative pair comparisons to isolate evidence.
"""
import sys,time,json
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];B=R/'experiments/astra_baseline';O=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'src'))
import features_v2 as F
import evaluate_v2 as E
keys=['src','eid'];t=time.time();scored=pl.read_parquet(B/'cache/dev_scored.parquet');nt=pl.read_parquet(B/'cache/ntrue_dev_20000_200.parquet');eligible=scored.filter(pl.col('score')>=.3).select('s1',*keys).unique();targets=eligible.select(keys).unique().with_row_index('qid');s1=pl.read_parquet(R/'work/cache/train_source1.parquet');pool=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{s}.parquet').join(targets.lazy().filter(pl.col('src')==s).select('eid'),on='eid',how='semi').collect(engine='streaming') for s in [2,3]]).join(targets,on=keys)
parts=[]
for field in ['name','addr']:
 counts=s1.filter(pl.col(field)!='').group_by(field).len();ref=s1.select(pl.col('eid').alias('s1'),field).join(counts.filter(pl.col('len')<=100).select(field),on=field,how='semi');parts.append(pool.select(*keys,field).join(ref,on=field).select('s1',*keys))
alt=pl.concat(parts).unique();allpairs=pl.concat([eligible,alt]).unique();print('eligible',eligible.height,'alternative pairs',alt.height,flush=True);model=lgb.Booster(model_file=str(B/'models/matcher_lgbm.txt'));outputs=[]
for start in range(0,allpairs.height,100000):
 q=allpairs.slice(start,100000);X=F.build(q,s1,pool).with_columns(pl.lit(100.,dtype=pl.Float32).alias('cand_count'));p=model.predict(X.select(F.FEATURES).to_numpy(),num_threads=2);outputs.append(X.select('s1',*keys).with_columns(pl.Series('fixed_count_score',p.astype('float32'))))
fixed=pl.concat(outputs);fixed.write_parquet(O/'reference_scores.parquet');current=eligible.join(fixed,on=['s1',*keys]);competitors=current.select(pl.col('s1').alias('owner_s1'),*keys).join(fixed.rename({'s1':'alternative_s1','fixed_count_score':'alternative_score'}),on=keys).filter(pl.col('owner_s1')!=pl.col('alternative_s1')).group_by('owner_s1',*keys).agg(pl.col('alternative_score').max().alias('alternative_score')).rename({'owner_s1':'s1'});aug=scored.join(current,on=['s1',*keys],how='left').join(competitors,on=['s1',*keys],how='left').with_columns(pl.col('alternative_score').fill_null(0.));rows=[]
for strong in [.9,.95,.99]:
 for margin in [.01,.05,.1,.2]:
  reject=(pl.col('alternative_score')>=strong)&(pl.col('alternative_score')>pl.col('fixed_count_score')+margin);filtered=aug.filter(~reject.fill_null(False));m=E.evaluate(filtered,nt,.62);m.update(strong=strong,margin=margin,rejected_pairs=aug.height-filtered.height,scope='dev-only diagnostic; full reference text, no alternative labels');rows.append(m)
pl.DataFrame(rows).write_csv(O/'ablation.csv');aug.filter(pl.col('alternative_score')>0).write_parquet(O/'competing_cases.parquet');print(pl.DataFrame(rows).sort('macro_f05',descending=True).head(5),flush=True);print('runtime',time.time()-t)
