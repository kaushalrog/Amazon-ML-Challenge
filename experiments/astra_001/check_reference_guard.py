import sys,polars as pl,numpy as np,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/astra_engine'));from reference_guard import ReferenceGuard
s=pl.read_parquet(R/'experiments/astra_011/broader_retrieval_dev_scored.parquet');ids=s.select('s1').unique().sort('s1').head(500);s=s.join(ids,on='s1',how='semi');ref=pl.read_parquet(R/'work/cache/train_source1.parquet');pool=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{i}.parquet').join(s.filter(pl.col('src')==i).select('eid').unique().lazy(),on='eid',how='semi').collect() for i in [2,3]]);guard=ReferenceGuard(ref);new=guard.enrich(s,pool);pieces=[]
for start in range(0,ids.height,50):pieces.append(guard.enrich(s.join(ids.slice(start,50),on='s1',how='semi'),pool))
other=pl.concat(pieces);joined=new.join(other,on=['s1','src','eid'],suffix='_old');out={c:joined.select((pl.col(c).fill_null(-1)-pl.col(c+'_old').fill_null(-1)).abs().max()).item() for c in ['fixed_count_score','alternative_score']};assert max(out.values())==0,out;print('PASS batch invariance',out)
(R/'reports/astra_reference_guard_equivalence.json').write_text(json.dumps(out,indent=2))
