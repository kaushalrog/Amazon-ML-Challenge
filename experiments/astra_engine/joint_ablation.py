from pathlib import Path
import sys,time
import polars as pl
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from key_index import Index
O=Path(__file__).parent;idx=Index('train');retr=pl.read_parquet(O/'retrieval_dev/part_000000000.parquet');ids=retr.select(pl.col('s1').alias('eid')).unique();q=pl.read_parquet(R/'work/cache/train_source1.parquet').join(ids,on='eid',how='semi');g=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(retr.select('s1').unique(),on='s1',how='semi');nt=ids.rename({'eid':'s1'}).join(g.group_by('s1').len().rename({'len':'ntrue'}),on='s1',how='left').fill_null(0);rows=[]
for cap in [200,400,800,1600]:
 t=time.time();count=hit=0;hs=[]
 for start in range(0,q.height,1000):
  z=q.slice(start,1000);c=pl.concat([idx.candidates(z,cap,True),retr.join(z.select(pl.col('eid').alias('s1')),on='s1',how='semi').select('s1','src','eid')]).unique();p=g.join(c,on=['s1','src','eid'],how='semi');count+=c.height;hit+=p.height;hs.append(p.group_by('s1').len().rename({'len':'hits'}))
 h=nt.join(pl.concat(hs),on='s1',how='left').fill_null(0);oracle=h.select(pl.when(pl.col('ntrue')==0).then(1.).otherwise(1.25*pl.col('hits')/(.25*pl.col('ntrue')+pl.col('hits')))).to_series().mean();row=dict(cap=cap,candidates=count,candidate_recall=hit/g.height,oracle_f05=oracle,runtime=time.time()-t);rows.append(row);print(row,flush=True);pl.DataFrame(rows).write_csv(O/'joint_ablation.csv')
