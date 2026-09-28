"""Memory-bounded full-corpus character retrieval with linear-time global top-K merge."""
import argparse,ctypes,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import polars as pl
from sparse_dot_topn import sp_matmul_topn
R=Path(__file__).resolve().parents[2];E=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--fold',choices=['train','dev','eval','fresh','test'],required=True);p.add_argument('--n',type=int);p.add_argument('--k',type=int,default=100);p.add_argument('--batch',type=int,default=60000);p.add_argument('--threads',type=int,default=2);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-only',action='store_true');a=p.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True);data='test' if a.fold=='test' else 'train';cache=R/'experiments/astra_004';mapping=json.loads((R/'experiments/astra_003/token_map.json').read_text());v={f:pickle.loads((R/f'experiments/astra_002/{f}_vectorizer.pkl').read_bytes()) for f in ['name','addr']}
def matrix(df,pool=False):
 names=df['name'].to_list()
 if pool:names=[' '.join(mapping.get(t,t) for t in text.split()) for text in names]
 return sp.hstack([v['name'].transform(names)*np.float32(.70710678),v['addr'].transform(df['addr'].to_list())*np.float32(.70710678)],format='csr')
paths=[R/f'work/cache/{data}_source{s}.parquet' for s in [2,3]];counts=[pl.scan_parquet(f).select(pl.len()).collect().item() for f in paths];shards=[];offset=0;t=time.time()
for src,path,total in zip([2,3],paths,counts):
 for start in range(0,total,250000):
  file=cache/f'{data}_s{src}_{start}.npz'
  if not file.exists():
   part=pl.scan_parquet(path).slice(start,250000).select('name','addr').collect();m=matrix(part,True).T.tocsr();tmp=file.with_name(file.stem+'.partial.npz');sp.save_npz(tmp,m,compressed=False);tmp.replace(file);print('built',file.name,round(time.time()-t),flush=True)
  shards.append((file,offset+start))
 offset+=total
if a.build_only:raise SystemExit(0)
if a.fold=='test':qall=pl.read_parquet(R/'work/cache/test_source1.parquet');qall=qall.head(a.n) if a.n else qall
else:
 if a.fold=='fresh':
  fold=pl.read_parquet(R/'experiments/astra_baseline/cache/folds.parquet').filter(pl.col('fold')=='eval');used=pl.read_parquet(R/'experiments/astra_baseline/cache/ntrue_eval_20000_200.parquet').select(pl.col('s1').alias('eid'));ids=fold.join(used,on='eid',how='anti').with_columns(pl.col('eid').hash(seed=20260928).alias('order')).sort('order').head(a.n or 20000).select('eid')
 else:ids=pl.read_parquet(R/f'experiments/astra_baseline/cache/ntrue_{a.fold}_{60000 if a.fold=="train" else 20000}_200.parquet').select(pl.col('s1').alias('eid')).sort('eid').head(a.n or (60000 if a.fold=='train' else 20000))
 qall=pl.scan_parquet(R/'work/cache/train_source1.parquet').join(ids.lazy(),on='eid',how='semi').collect().sort('eid')
qall.select('eid').write_parquet(O/'query_ids.parquet');signature={'fold':a.fold,'n':qall.height,'k':a.k,'batch':a.batch,'model_inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [R/'experiments/astra_003/token_map.json',*[R/f'experiments/astra_002/{field}_vectorizer.pkl' for field in ['name','addr']]]}}
if (O/'manifest.json').exists():assert json.loads((O/'manifest.json').read_text())==signature,'Retrieval run mismatch'
else:(O/'manifest.json').write_text(json.dumps(signature,indent=2))
eids=np.concatenate([pl.read_parquet(f,columns=['eid'])['eid'].to_numpy() for f in paths]);lib=ctypes.CDLL(str(E/'merge_topk.dylib'));merge=lib.merge_topk;merge.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ctypes.c_int32,ctypes.c_int32,ctypes.c_int32];merge.restype=None
for start in range(0,qall.height,a.batch):
 file=O/f'part_{start:09d}.parquet'
 if file.exists():print('resume skip',start,flush=True);continue
 q=qall.slice(start,a.batch);qm=matrix(q);best=np.full((q.height,a.k),-np.inf,dtype='float32');bestids=np.full((q.height,a.k),-1,dtype='int32');tt=time.time()
 for j,(path,offset) in enumerate(shards):
  m=sp.load_npz(path);z=sp_matmul_topn(qm,m,top_n=a.k,threshold=.1,sort=True,n_threads=a.threads);assert z.indices.dtype==z.indptr.dtype==np.int32;merge(best,bestids,z.data,z.indices,z.indptr,q.height,a.k,offset)
  if j%10==0:print('batch',start,'shard',j+1,len(shards),'seconds',round(time.time()-tt),flush=True)
 mask=bestids.ravel()>=0;pos=bestids.ravel()[mask];out=pl.DataFrame({'s1':np.repeat(q['eid'].to_numpy(),a.k)[mask],'src':np.where(pos<counts[0],2,3).astype('uint8'),'eid':eids[pos],'joint_retrieval':best.ravel()[mask]});tmp=file.with_suffix('.tmp');out.write_parquet(tmp);tmp.replace(file);print('complete batch',start,'pairs',out.height,'seconds',round(time.time()-tt),flush=True)
print('DONE seconds',time.time()-t,flush=True)
