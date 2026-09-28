"""Incremental feature cache and full-denominator development scoring of real candidates."""
from pathlib import Path
import argparse,sys,json,time,importlib.util
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];B=R/'experiments/astra_baseline';sys.path.insert(0,str(B/'src'))
import features_v2 as F
import evaluate_v2 as E
spec=importlib.util.spec_from_file_location('astra_extra',R/'experiments/astra_003/features.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=2000);p.add_argument('--model',type=Path,default=B/'models/matcher_lgbm.txt');p.add_argument('--decision',type=Path,default=B/'models/decision.json');args=p.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True);t=time.time();nt=pl.read_parquet(B/'cache/ntrue_dev_20000_200.parquet').sort('s1').head(args.n);keys=['s1','src','eid'];cand=pl.concat([pl.read_parquet(x).select(keys) for x in args.candidate]).unique();assert cand.join(nt.select('s1'),on='s1',how='anti').height==0;counts=cand.group_by('s1').len().select('s1',pl.col('len').cast(pl.Float32).alias('cand_count'))
truth=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(nt.select('s1'),on='s1',how='semi');config=json.loads(args.decision.read_text());features=config['features'];model=lgb.Booster(model_file=str(args.model))
if (O/'features.parquet').exists():
 X=pl.read_parquet(O/'features.parquet');assert X.select(keys).sort(keys).equals(cand.sort(keys)),'Candidate cache mismatch'
else:
 old=pl.read_parquet(R/'experiments/astra_005/pilot_features.parquet').join(pl.read_parquet(R/'experiments/astra_005/pilot_extra.parquet'),on=keys,validate='1:1').join(cand,on=keys,how='semi');new=cand.join(old.select(keys),on=keys,how='anti');print('reuse',old.height,'new',new.height,flush=True)
 chunks=[]
 if new.height:
  s1=pl.read_parquet(R/'work/cache/train_source1.parquet').join(nt.select(pl.col('s1').alias('eid')),on='eid',how='semi');pool=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{s}.parquet').join(new.lazy().filter(pl.col('src')==s).select('eid').unique(),on='eid',how='semi').collect(engine='streaming') for s in [2,3]])
  for start in range(0,new.height,100000):
   j=F.attach_records(new.slice(start,100000),s1,pool);z=F.compute(j).join(A.compute(j,A.resources()),on=keys,validate='1:1').join(truth.with_columns(pl.lit(1,dtype=pl.Int8).alias('label')),on=keys,how='left').with_columns(pl.col('label').fill_null(0));chunks.append(z.select(old.columns));print('features',start+z.height,flush=True)
 X=pl.concat([old,*chunks]).drop('cand_count').join(counts,on='s1');X.write_parquet(O/'features.parquet')
s=X.select(*keys,'label').with_columns(pl.Series('score',model.predict(X.select(features).to_numpy(),num_threads=2).astype('float32')));s.write_parquet(O/'scored.parquet');m=E.evaluate(s,nt,config['threshold']);sweep=E.sweep(s,nt,np.round(np.arange(.30,.951,.025),3));sweep.write_csv(O/'thresholds.csv');base=pl.read_parquet(B/'cache/dev_scored.parquet').join(nt.select('s1'),on='s1',how='semi');bm=E.evaluate(base,nt,.62);m.update(candidate_recall=X['label'].sum()/truth.height,candidate_count=X.height,baseline_same_entities=bm['macro_f05'],gain=m['macro_f05']-bm['macro_f05'],runtime=time.time()-t,model=str(args.model),status='development pilot; threshold frozen from model manifest');(O/'results.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True)
