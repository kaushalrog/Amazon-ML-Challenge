from pathlib import Path
import argparse,json,sys,time
import polars as pl
from reference_guard import ReferenceGuard,accepted,R,B,KEY
sys.path.insert(0,str(B/'src'));import evaluate_v2 as E
p=argparse.ArgumentParser();p.add_argument('--scored',type=Path,required=True);p.add_argument('--ntrue',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True);s=pl.read_parquet(a.scored);s=s.drop([c for c in ['fixed_count_score','alternative_score'] if c in s.columns]);nt=pl.read_parquet(a.ntrue);cfg=json.loads(a.decision.read_text());eligible=s.filter(pl.col('score')>=.02);pool=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{i}.parquet').join(eligible.filter(pl.col('src')==i).select('eid').unique().lazy(),on='eid',how='semi').collect() for i in [2,3]]);t=time.time();aug=ReferenceGuard(pl.read_parquet(R/'work/cache/train_source1.parquet')).enrich(s,pool);aug.write_parquet(O/'scored.parquet');rows=[]
for strong in [.9,.95,.99]:
 for margin in [.01,.05,.1,.2]:
  cc=dict(cfg);cc['reference_guard']={'strong':strong,'margin':margin};sel=accepted(aug,cc);m=E.evaluate(sel,nt,0.);m.update(strong=strong,margin=margin,threshold=cfg['threshold']);rows.append(m)
pl.DataFrame(rows).sort('macro_f05',descending=True).write_csv(O/'guard_ablation.csv');print(pl.DataFrame(rows).sort('macro_f05',descending=True).head(4).to_dicts(),flush=True);print('seconds',time.time()-t,flush=True)
