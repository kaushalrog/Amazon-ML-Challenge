"""Ablate a cheap rejection gate AFTER context features, before expensive inference.

Context and raw candidate counts remain defined by the original first-stage set.
"""
from pathlib import Path
import argparse,sys,json,time
import polars as pl,numpy as np,lightgbm as lgb
from prepare import R,B,KEY
from reference_guard import accepted
sys.path.insert(0,str(B/'src'));import evaluate_v2 as E
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--scored',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=json.loads(a.decision.read_text());x=pl.read_parquet(a.prepared/'parts/*.parquet');nt=pl.read_parquet(a.prepared/'ntrue.parquet');m=lgb.Booster(model_file=str(R/'experiments/astra_fast/blocker.txt'));features=json.loads((R/'experiments/astra_fast/manifest.json').read_text())['features'];fast=x.select(KEY).with_columns(pl.Series('late_gate_score',m.predict(x.select(features).to_numpy(order='c'),num_threads=2).astype('float32')));fast.write_parquet(a.output/'gate_scores.parquet');s=pl.read_parquet(a.scored).join(fast,on=KEY,validate='1:1');rows=[]
for threshold in [.0001,.0005,.001,.002,.005,.01,.02,.05,.1]:
 z=s.filter(pl.col('late_gate_score')>=threshold);sel=accepted(z,cfg);out=E.evaluate(sel,nt,0.);out.update(gate_threshold=threshold,expensive_pairs=z.height,effective_pair_recall=z['label'].sum()/nt['ntrue'].sum());rows.append(out)
pl.DataFrame(rows).write_csv(a.output/'ablation.csv');print(pl.DataFrame(rows).select('gate_threshold','macro_f05','expensive_pairs','effective_pair_recall','tp','false_positives').to_dicts(),flush=True)
