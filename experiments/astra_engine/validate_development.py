"""Complete development or untouched holdout evaluation, with frozen model and real blocking."""
from pathlib import Path
import argparse,json,sys,time,importlib.util
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];B=R/'experiments/astra_baseline';sys.path.insert(0,str(B/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from key_index import Index
import features_v2 as F
import evaluate_v2 as E
spec=importlib.util.spec_from_file_location('astra_extra',R/'experiments/astra_003/features.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cap',type=int,default=200);p.add_argument('--fresh',action='store_true');p.add_argument('--n',type=int,default=20000);p.add_argument('--extra-blocks',action='store_true');args=p.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True);cfg=json.loads(args.decision.read_text());features=cfg['features'];model=lgb.Booster(model_file=str(args.model));index=Index('train');res=A.resources();t=time.time()
if args.fresh:
 fold=pl.read_parquet(B/'cache/folds.parquet').filter(pl.col('fold')=='eval');used=pl.read_parquet(B/'cache/ntrue_eval_20000_200.parquet').select(pl.col('s1').alias('eid'));ids=fold.join(used,on='eid',how='anti').with_columns(pl.col('eid').hash(seed=20260928).alias('order')).sort('order').head(args.n).select('eid')
else:ids=pl.read_parquet(B/'cache/ntrue_dev_20000_200.parquet').select(pl.col('s1').alias('eid')).sort('eid').head(args.n)
s1=pl.scan_parquet(R/'work/cache/train_source1.parquet').join(ids.lazy(),on='eid',how='semi').collect();truth=pl.scan_parquet(R/'work/cache/gt_pairs.parquet').join(ids.lazy().rename({'eid':'s1'}),on='s1',how='semi').collect();nt=ids.rename({'eid':'s1'}).join(truth.group_by('s1').len().rename({'len':'ntrue'}),on='s1',how='left').fill_null(0);nt.write_parquet(O/'ntrue.parquet');parts=O/'parts';parts.mkdir(exist_ok=True)
pool=pl.concat([pl.read_parquet(R/f'work/cache/train_source{s}.parquet') for s in [2,3]]);pred=[];hit=count=0
for start in range(0,s1.height,1000):
 file=parts/f'{start:06d}.parquet'
 if file.exists():ss=pl.read_parquet(file)
 else:
  q=s1.slice(start,1000);cand=index.candidates(q,args.cap,extra=args.extra_blocks);joined=F.attach_records(cand,q,pool);X=F.compute(joined)
  if any(f in A.EXTRA for f in features):X=X.join(A.compute(joined,res),on=['s1','src','eid'],validate='1:1')
  X=X.join(truth.with_columns(pl.lit(1,dtype=pl.Int8).alias('label')),on=['s1','src','eid'],how='left').with_columns(pl.col('label').fill_null(0));ss=X.select('s1','src','eid','label').with_columns(pl.Series('score',model.predict(X.select(features).to_numpy(),num_threads=2).astype('float32')));ss.write_parquet(file)
 pred.append(ss);hit+=ss['label'].sum();count+=ss.height;print('scored',start+min(1000,s1.height-start),'candidates',count,'seconds',round(time.time()-t),flush=True)
s=pl.concat(pred);s.write_parquet(O/'scored.parquet');m=E.evaluate(s,nt,cfg['threshold']);m.update(candidate_recall=hit/truth.height,candidate_count=count,cap=args.cap,extra_blocks=args.extra_blocks,fresh_holdout=args.fresh,runtime=time.time()-t,model=str(args.model));(O/'metrics.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True)
if not args.fresh:E.sweep(s,nt,np.round(np.arange(.3,.951,.025),3)).write_csv(O/'thresholds.csv')
