from pathlib import Path
import sys,time
import numpy as np,polars as pl
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R/'experiments/astra_engine'));from prepare import F,KEY
import expanded_map as M
m=M.resources();left=pl.read_parquet(R/'work/cache/train_source1.parquet');pool=pl.concat([pl.read_parquet(R/f'work/cache/train_source{s}.parquet') for s in [2,3]]).with_columns((pl.col('eid')*4+pl.col('src')).alias('_key')).sort('_key');pk=pool['_key'].to_numpy();t=time.time()
for fold in ['train','dev','fresh']:
 file=O/f'{fold}_extra.parquet'
 if file.exists():continue
 x=pl.read_parquet(R/'experiments/astra_011/mined_train.parquet',columns=KEY) if fold=='train' else pl.read_parquet(R/f'experiments/astra_011/{fold}/parts/*.parquet',columns=KEY);out=[]
 for start in range(0,x.height,100000):
  c=x.slice(start,100000);wanted=np.unique(c['eid'].to_numpy()*4+c['src'].to_numpy());pos=np.searchsorted(pk,wanted);assert np.array_equal(pk[pos],wanted);j=F.attach_records(c,left,pool[pos]);out.append(M.compute(j,m))
 pl.concat(out).write_parquet(file);print(fold,x.height,'seconds',round(time.time()-t),flush=True)
