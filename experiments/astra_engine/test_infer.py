"""Resume complete test inference as atomic retrieval batches become available."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import polars as pl
import lightgbm as lgb
from prepare import R,B,E,F,A,KEY,Index
from infer import write_rows
import context as C,expanded_map as M
from reference_guard import ReferenceGuard,accepted
def main():
 p=argparse.ArgumentParser();p.add_argument('--retrieval',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=500);p.add_argument('--wait',action='store_true');a=p.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True);parts=O/'parts';parts.mkdir(exist_ok=True);cfg=json.loads(a.decision.read_text());features=cfg['features'];gate_model=lgb.Booster(model_file=str(R/cfg['full_feature_gate']['model'])) if 'full_feature_gate' in cfg else None;model=lgb.Booster(model_file=str(a.model));assert model.feature_name()==features
 sig={'configuration':cfg,'batch':a.batch,'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),a.model,a.decision,E/'context.py',E/'expanded_map.py',E/'reference_guard.py',E/'probabilistic_decision.py',E/'expected_f05.cpp',E/'expected_f05.dylib',R/'experiments/astra_013/token_map.json',R/'experiments/astra_final/full_feature_gate.txt',B/'models/matcher_lgbm.txt',E/'key_index.py',B/'src/features_v2.py',R/'experiments/astra_003/features.py',R/'experiments/astra_003/token_map.json',R/'experiments/astra_003/name_df.json',R/'experiments/astra_fast/blocker.txt',a.retrieval/'manifest.json']}}
 manifest=O/'run_manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())==sig,'Inference manifest mismatch'
 else:manifest.write_text(json.dumps(sig,indent=2))
 qall=pl.read_parquet(R/'work/cache/test_source1.parquet');assert qall.select('eid').equals(pl.read_parquet(a.retrieval/'query_ids.parquet')),'Retrieval query order differs from raw test order';rconf=json.loads((a.retrieval/'manifest.json').read_text());needmap=any(f in M.FEATURES for f in features);mapres=M.resources() if needmap else None;guard=ReferenceGuard(qall) if 'reference_guard' in cfg else None;idx=Index('test');res=A.resources();fast=lgb.Booster(model_file=str(R/'experiments/astra_fast/blocker.txt'));ff=json.loads((R/'experiments/astra_fast/manifest.json').read_text())['features'];pool=pl.concat([pl.read_parquet(R/f'work/cache/test_source{s}.parquet') for s in [2,3]]).with_columns((pl.col('eid')*4+pl.col('src')).alias('_key')).sort('_key');pk=pool['_key'].to_numpy();t=time.time()
 for rstart in range(0,qall.height,rconf['batch']):
  rf=a.retrieval/f'part_{rstart:09d}.parquet'
  while not rf.exists():
   if not a.wait:raise RuntimeError(f'Missing retrieval checkpoint {rf}')
   time.sleep(10)
  retrieval=pl.read_parquet(rf);q=qall.slice(rstart,rconf['batch'])
  for local in range(0,q.height,a.batch):
   start=rstart+local;prefix=parts/f'{start:09d}';done=prefix.with_suffix('.json');mp=prefix.with_suffix('.matching.tsv');cp=prefix.with_suffix('.candidate.tsv')
   if done.exists() and mp.exists() and cp.exists():continue
   z=q.slice(local,a.batch);rr=retrieval.join(z.select(pl.col('eid').alias('s1')),on='s1',how='semi');c=pl.concat([idx.candidates(z,cfg['candidate_cap'],True),rr.select(KEY)]).unique().join(rr,on=KEY,how='left').with_columns(pl.col('joint_retrieval').fill_null(0));raw=c.height
   if raw:
    wanted=np.unique(c['eid'].to_numpy()*4+c['src'].to_numpy());pos=np.searchsorted(pk,wanted);assert np.all(pos<len(pk)) and np.array_equal(pk[pos],wanted);j=F.attach_records(c,z,pool[pos]);X=F.compute(j);fp=fast.predict(X.select(ff).to_numpy(),num_threads=2);keep=X.select(KEY).filter(pl.Series(fp>=cfg['fast_filter']));X=X.join(keep,on=KEY,how='semi')
    if X.height:
     jj=j.join(keep,on=KEY,how='semi');X=X.join(A.compute(jj,res),on=KEY,validate='1:1').join(c.select(*KEY,'joint_retrieval'),on=KEY,validate='1:1')
     if needmap:X=X.join(M.compute(jj,mapres),on=KEY,validate='1:1')
     X=C.add(X);arr=X.select(features).to_numpy(order='c');gp=gate_model.predict(arr,num_threads=2).astype('float32') if gate_model is not None else np.ones(X.height,dtype='float32');eligible=gp>=cfg.get('full_feature_gate',{}).get('threshold',0.);score=np.zeros(X.height,dtype='float32')
     if eligible.any():score[eligible]=model.predict(arr[eligible],num_threads=2).astype('float32')
     scored=X.select(KEY).with_columns(pl.Series('score',score),pl.Series('gate_score',gp))
    else:scored=keep.with_columns(pl.lit(0.,dtype=pl.Float32).alias('score'))
   else:scored=c.select(KEY).with_columns(pl.lit(0.,dtype=pl.Float32).alias('score'))
   if 'gate_score' not in scored.columns:scored=scored.with_columns(pl.lit(0.,dtype=pl.Float32).alias('gate_score'))
   if guard is not None and scored.height:scored=guard.enrich(scored,pool[pos])
   sel=accepted(scored,cfg);tc=cp.with_suffix('.tmp');tm=mp.with_suffix('.tmp');write_rows(tc,z,scored,'candidate_entity_ids');empty=write_rows(tm,z,sel,'matched_entity_ids');tc.replace(cp);tm.replace(mp)
   st={'start':start,'rows':z.height,'raw_candidates':raw,'candidates':scored.height,'predictions':sel.height,'empty':empty};done.write_text(json.dumps(st));
   if local%5000==0:print(st,'seconds',round(time.time()-t),flush=True)
  print('completed retrieval batch',rstart,'seconds',round(time.time()-t),flush=True)
 stats=[json.loads(f.read_text()) for f in sorted(parts.glob('*.json'))];assert sum(s['rows'] for s in stats)==qall.height
 for kind,header,filename in [('matching','matched_entity_ids','matching_results.tsv'),('candidate','candidate_entity_ids','candidate_pairs.tsv')]:
  target=O/filename;tmp=target.with_suffix('.tmp')
  with tmp.open('w') as out:
   out.write('source1_entity_id\t'+header+'\n')
   for f in sorted(parts.glob(f'*.{kind}.tsv')):
    with f.open() as inp:
     next(inp)
     for line in inp:out.write(line)
  tmp.replace(target)
 result={k:sum(s[k] for s in stats) for k in ['rows','raw_candidates','candidates','predictions','empty']};result['elapsed_this_invocation']=time.time()-t;(O/'output_stats.json').write_text(json.dumps(result,indent=2));print('DONE',result,flush=True)
if __name__=='__main__':main()
