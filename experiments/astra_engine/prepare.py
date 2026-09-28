"""Real blocker -> conservative fast filter -> train-fitted pair features.

Labels are attached only after candidate generation and filtering. Checkpoints
are bound to query IDs, fitted artifacts, configuration and feature code.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,importlib.util
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];B=R/'experiments/astra_baseline';E=Path(__file__).parent
sys.path.insert(0,str(B/'src'));sys.path.insert(0,str(E))
from key_index import Index
import features_v2 as F
spec=importlib.util.spec_from_file_location('astra_extra',R/'experiments/astra_003/features.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
KEY=['s1','src','eid']
def main():
 p=argparse.ArgumentParser();p.add_argument('--retrieval',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cap',type=int,required=True);p.add_argument('--filter',type=float,default=.0001);p.add_argument('--batch',type=int,default=500);p.add_argument('--test',action='store_true');a=p.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True);parts=O/'parts';parts.mkdir(exist_ok=True);split='test' if a.test else 'train';t=time.time()
 files=[Path(__file__),E/'key_index.py',B/'src/features_v2.py',R/'experiments/astra_003/features.py',R/'experiments/astra_003/token_map.json',R/'experiments/astra_003/name_df.json',R/'experiments/astra_fast/blocker.txt',a.retrieval/'query_ids.parquet',a.retrieval/'manifest.json']
 signature={'cap':a.cap,'filter':a.filter,'batch':a.batch,'split':split,'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
 if (O/'manifest.json').exists():assert json.loads((O/'manifest.json').read_text())==signature,'Preparation cache configuration mismatch'
 else:(O/'manifest.json').write_text(json.dumps(signature,indent=2))
 ids=pl.read_parquet(a.retrieval/'query_ids.parquet');qall=ids.join(pl.read_parquet(R/f'work/cache/{split}_source1.parquet'),on='eid',maintain_order='left');idx=Index(split);res=A.resources();fast=lgb.Booster(model_file=str(R/'experiments/astra_fast/blocker.txt'));ff=json.loads((R/'experiments/astra_fast/manifest.json').read_text())['features']
 pool=pl.concat([pl.read_parquet(R/f'work/cache/{split}_source{s}.parquet') for s in [2,3]])
 # Binary search gives bounded record gathers, avoiding a full corpus hash join per batch.
 pool=pool.with_columns((pl.col('eid')*4+pl.col('src')).alias('_key')).sort('_key');pk=pool['_key'].to_numpy()
 truth=None
 if not a.test:
  truth=pl.scan_parquet(R/'work/cache/gt_pairs.parquet').join(ids.lazy().rename({'eid':'s1'}),on='s1',how='semi').collect();nt=ids.rename({'eid':'s1'}).join(truth.group_by('s1').len().rename({'len':'ntrue'}),on='s1',how='left').fill_null(0);nt.write_parquet(O/'ntrue.parquet')
 for rf in sorted(a.retrieval.glob('part_*.parquet')):
  retrieval=pl.read_parquet(rf);rids=retrieval.select('s1').unique();q=qall.join(rids.rename({'s1':'eid'}),on='eid',how='semi',maintain_order='left')
  for start in range(0,q.height,a.batch):
   tag=rf.stem+f'_{start:06d}';out=parts/(tag+'.parquet');stats=parts/(tag+'.json')
   if out.exists() and stats.exists():continue
   z=q.slice(start,a.batch);rr=retrieval.join(z.select(pl.col('eid').alias('s1')),on='s1',how='semi');c=pl.concat([idx.candidates(z,a.cap,True),rr.select(KEY)]).unique().join(rr,on=KEY,how='left').with_columns(pl.col('joint_retrieval').fill_null(0));raw=c.height
   wanted=np.unique(c['eid'].to_numpy()*4+c['src'].to_numpy());pos=np.searchsorted(pk,wanted);assert np.all(pos<len(pk)) and np.array_equal(pk[pos],wanted)
   j=F.attach_records(c,z,pool[pos]);X=F.compute(j);fp=fast.predict(X.select(ff).to_numpy(),num_threads=2);keep=X.select(KEY).filter(pl.Series(fp>=a.filter));X=X.join(keep,on=KEY,how='semi');jj=j.join(keep,on=KEY,how='semi');X=X.join(A.compute(jj,res),on=KEY,validate='1:1').join(c.select(*KEY,'joint_retrieval'),on=KEY,validate='1:1')
   report={'s1_count':z.height,'raw_candidates':raw,'retained_candidates':X.height,'seconds_cumulative':time.time()-t}
   if truth is not None:
    gt=truth.join(z.select(pl.col('eid').alias('s1')),on='s1',how='semi');report.update(true_pairs=gt.height,raw_hits=gt.join(c,on=KEY,how='semi').height,retained_hits=gt.join(X,on=KEY,how='semi').height);X=X.join(gt.with_columns(pl.lit(1,dtype=pl.Int8).alias('label')),on=KEY,how='left').with_columns(pl.col('label').fill_null(0))
   tmp=out.with_suffix('.tmp');X.write_parquet(tmp);tmp.replace(out);stats.write_text(json.dumps(report,indent=2));print(tag,report,flush=True)
 print('DONE',time.time()-t,flush=True)
if __name__=='__main__':main()
