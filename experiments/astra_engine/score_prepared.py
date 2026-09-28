"""Evaluate a frozen pipeline using full S1 truth denominators; optional dev-only sweep."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np,polars as pl,lightgbm as lgb
from prepare import R,B,E,F,KEY
import context as C,expanded_map as M
from reference_guard import ReferenceGuard,accepted
sys.path.insert(0,str(B/'src'));import evaluate_v2 as V
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dev',action='store_true');p.add_argument('--enrich-guard',action='store_true');a=p.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True);t=time.time();cfg=json.loads(a.decision.read_text());x=C.add(pl.read_parquet(a.prepared/'parts/*.parquet'));nt=pl.read_parquet(a.prepared/'ntrue.parquet');needmap=any(f in M.FEATURES for f in cfg['features']);guard=a.enrich_guard or 'reference_guard' in cfg
if needmap or guard:
 ref=pl.read_parquet(R/'work/cache/train_source1.parquet');pool=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{s}.parquet').join(x.filter(pl.col('src')==s).select('eid').unique().lazy(),on='eid',how='semi').collect() for s in [2,3]]).with_columns((pl.col('eid')*4+pl.col('src')).alias('_key')).sort('_key');pk=pool['_key'].to_numpy()
if needmap:
 mapping=M.resources();extras=[]
 for start in range(0,x.height,100000):
  c=x.slice(start,100000).select(KEY);wanted=np.unique(c['eid'].to_numpy()*4+c['src'].to_numpy());pos=np.searchsorted(pk,wanted);assert np.array_equal(pk[pos],wanted);extras.append(M.compute(F.attach_records(c,ref,pool[pos]),mapping))
 x=x.join(pl.concat(extras),on=KEY,validate='1:1')
model=lgb.Booster(model_file=str(a.model));assert model.feature_name()==cfg['features'];s=x.select(*KEY,'label').with_columns(pl.Series('score',model.predict(x.select(cfg['features']).to_numpy(),num_threads=2).astype('float32')))
if 'full_feature_gate' in cfg:
 gm=lgb.Booster(model_file=str(R/cfg['full_feature_gate']['model']));s=s.with_columns(pl.Series('gate_score',gm.predict(x.select(cfg['features']).to_numpy(order='c'),num_threads=2).astype('float32')))
if guard:s=ReferenceGuard(ref).enrich(s,pool)
s.write_parquet(O/'scored.parquet');selected=accepted(s,cfg);m=V.evaluate(selected,nt,0.);m.update(threshold=cfg['threshold'],candidate_recall=x['label'].sum()/nt['ntrue'].sum(),candidate_count=x.height,runtime=time.time()-t,model=str(a.model),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),decision=cfg,evaluation_type='development' if a.dev else 'reserved holdout');
if 'full_feature_gate' in cfg:
 unc=dict(cfg);unc.pop('full_feature_gate');m['ungated_macro_f05']=V.evaluate(accepted(s,unc),nt,0.)['macro_f05']
(O/'metrics.json').write_text(json.dumps(m,indent=2));print({k:v for k,v in m.items() if k!='decision'},flush=True)
if a.dev:V.sweep(s,nt,np.round(np.arange(.3,.951,.025),3)).write_csv(O/'thresholds.csv')
