"""Full development blocking ablation using bounded S1 batches and complete truth."""
from pathlib import Path
import sys,json,time
import polars as pl
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent));from key_index import Index,KEYS
B=R/'experiments/astra_baseline';O=R/'experiments/astra_engine';index=Index('train');assert set(KEYS)|{'rare_name','rare_addr'} <= set(index.keys),'Incomplete blocking index'
nt=pl.read_parquet(B/'cache/ntrue_dev_20000_200.parquet');s1=pl.scan_parquet(R/'work/cache/train_source1.parquet').join(nt.lazy().select(pl.col('s1').alias('eid')),on='eid',how='semi').collect();truth=pl.scan_parquet(R/'work/cache/gt_pairs.parquet').join(nt.lazy().select('s1'),on='s1',how='semi').collect();base=pl.read_parquet(B/'cache/dev_scored.parquet').select('s1','src','eid');rows=[]
for extra,cap in [(False,200),(True,200),(True,400),(True,800),(True,1600)]:
 t=time.time();count=hits=max_count=0;oracle=[]
 for start in range(0,s1.height,1000):
  q=s1.slice(start,1000);c=index.candidates(q,cap,extra=extra);ids=q.select(pl.col('eid').alias('s1'));g=truth.join(ids,on='s1',how='semi');positive=g.join(c,on=['s1','src','eid'],how='semi');h=nt.join(ids,on='s1',how='semi').join(positive.group_by('s1').len().rename({'len':'hits'}),on='s1',how='left').fill_null(0);oracle.extend(h.select(pl.when(pl.col('ntrue')==0).then(1.).otherwise(1.25*pl.col('hits')/(.25*pl.col('ntrue')+pl.col('hits'))).alias('f'))['f'].to_list())
  if not extra:
   expected=base.join(ids,on='s1',how='semi');assert c.sort('s1','src','eid').equals(expected.sort('s1','src','eid')),'Indexed baseline differs from protected candidate set'
  count+=c.height;hits+=positive.height;max_count=max(max_count,c.group_by('s1').len()['len'].max() or 0)
 row={'extra_blocks':extra,'cap':cap,'n_s1':s1.height,'candidates':count,'candidate_recall':hits/truth.height,'avg_candidates':count/s1.height,'max_candidates':max_count,'oracle_macro_f05':float(np.mean(oracle)),'runtime':time.time()-t};rows.append(row);pl.DataFrame(rows).write_csv(O/'blocking_ablation.csv');print(row,flush=True)
