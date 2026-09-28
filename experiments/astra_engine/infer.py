"""Versioned, resumable inference with exact candidate/model/output correspondence.

Requires a completed disk index and a frozen model manifest. The manifest's
feature names are authoritative; unsupported features fail before output writing.
"""
import argparse,json,time,sys,hashlib,importlib.util
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/astra_baseline/src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from key_index import Index
import features_v2 as F
spec=importlib.util.spec_from_file_location('astra_extra',R/'experiments/astra_003/features.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def write_rows(path,base,pairs,column):
 ids=pairs.select('s1',pl.concat_str(pl.lit('S'),pl.col('src').cast(pl.String),pl.lit('-'),pl.col('eid').cast(pl.String)).alias('id')).group_by('s1').agg(pl.col('id').sort().str.join(','))
 rows=base.select(pl.col('eid').alias('s1')).join(ids,on='s1',how='left',maintain_order='left').with_columns(pl.col('id').fill_null('')).select((pl.lit('S1-')+pl.col('s1').cast(pl.String)).alias('source1_entity_id'),pl.col('id').alias(column))
 rows.write_csv(path,separator='\t',quote_style='never');return rows.filter(pl.col(column)=='').height

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extra-blocks',action='store_true');p.add_argument('--cap',type=int,default=200);p.add_argument('--batch-size',type=int,default=2000);p.add_argument('--limit',type=int);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);parts=out/'parts';parts.mkdir(exist_ok=True)
 config=json.loads(args.decision.read_text());features=config['features'];threshold=config['threshold'];assert set(features)<=set(F.FEATURES+A.EXTRA),'Unimplemented model features';model=lgb.Booster(model_file=str(args.model));assert model.num_feature()==len(features),'Model/schema length mismatch';assert model.feature_name() in (features,[f'Column_{i}' for i in range(len(features))]),'Model/schema names mismatch'
 sig={'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'decision':config,'extra_blocks':args.extra_blocks,'cap':args.cap,'batch_size':args.batch_size,'limit':args.limit,'feature_code_sha256':hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest(),'inference_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 manifest=out/'run_manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())==sig,'Refusing to mix incompatible inference runs'
 else:manifest.write_text(json.dumps(sig,indent=2))
 index=Index('test');s1all=pl.read_parquet(R/'work/cache/test_source1.parquet');s1all=s1all.head(args.limit) if args.limit else s1all
 pool=pl.concat([pl.read_parquet(R/f'work/cache/test_source{s}.parquet') for s in [2,3]]);resources=A.resources();t=time.time();stats=[]
 for start in range(0,s1all.height,args.batch_size):
  prefix=parts/f'{start:09d}';mp=prefix.with_suffix('.matching.tsv');cp=prefix.with_suffix('.candidate.tsv');done=prefix.with_suffix('.json')
  if done.exists():stats.append(json.loads(done.read_text()));continue
  q=s1all.slice(start,args.batch_size);c=index.candidates(q,args.cap,extra=args.extra_blocks);x=c
  if c.height:
   joined=F.attach_records(c,q,pool);x=F.compute(joined)
   if any(f in A.EXTRA for f in features):x=x.join(A.compute(joined,resources),on=['s1','src','eid'],validate='1:1')
  if x.height:p=model.predict(x.select(features).to_numpy(),num_threads=2);scored=x.select('s1','src','eid').with_columns(pl.Series('score',p))
  else:scored=c.with_columns(pl.lit(0.).alias('score'))
  selected=scored.filter(pl.col('score')>=threshold).select('s1','src','eid');tmpc=cp.with_suffix('.tmp');tmpm=mp.with_suffix('.tmp');write_rows(tmpc,q,scored.select('s1','src','eid'),'candidate_entity_ids');empty=write_rows(tmpm,q,selected,'matched_entity_ids');tmpc.replace(cp);tmpm.replace(mp)
  st={'start':start,'rows':q.height,'candidates':scored.height,'predictions':selected.height,'empty':empty};done.write_text(json.dumps(st));stats.append(st);print(st,'elapsed',round(time.time()-t),flush=True)
 for kind,header in [('matching','matched_entity_ids'),('candidate','candidate_entity_ids')]:
  final=out/('matching_results.tsv' if kind=='matching' else 'candidate_pairs.tsv');tmp=final.with_suffix('.tmp')
  with tmp.open('w') as target:
   target.write('source1_entity_id\t'+header+'\n')
   for start in range(0,s1all.height,args.batch_size):
    with (parts/f'{start:09d}.{kind}.tsv').open() as source:
     next(source)
     for line in source:target.write(line)
  tmp.replace(final)
 result={k:sum(s[k] for s in stats) for k in ['rows','candidates','predictions','empty']};result['elapsed_seconds_this_invocation']=time.time()-t;(out/'output_stats.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
