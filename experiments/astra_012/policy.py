"""Dev-only per-source threshold ablation, preserving multiple accepted matches."""
from pathlib import Path
import argparse,json,sys,time
import polars as pl,numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/astra_baseline/src'));import evaluate_v2 as E
p=argparse.ArgumentParser();p.add_argument('--scored',type=Path,required=True);p.add_argument('--ntrue',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);s=pl.read_parquet(a.scored);nt=pl.read_parquet(a.ntrue);cfg=json.loads(a.decision.read_text());base=E.evaluate(s,nt,cfg['threshold']);rows=[]
for t2 in np.round(np.arange(max(.3,cfg['threshold']-.1),min(.99,cfg['threshold']+.101),.025),3):
 for t3 in np.round(np.arange(max(.3,cfg['threshold']-.1),min(.99,cfg['threshold']+.101),.025),3):
  q=s.filter(pl.col('score')>=pl.when(pl.col('src')==2).then(t2).otherwise(t3));m=E.evaluate(q,nt,0.);m.update(t2=t2,t3=t3,gain=m['macro_f05']-base['macro_f05']);rows.append(m)
pl.DataFrame(rows).sort('macro_f05',descending=True).write_csv(a.output/'source_thresholds.csv');print(pl.DataFrame(rows).sort('macro_f05',descending=True).head(5))
