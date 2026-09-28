"""Ablate sparse retrieval retaining each record's strongest character evidence.

Pruning is candidate generation only, never a match decision. It is evaluated
against all truth pairs, with full sparse retrieval as the control.
"""
from pathlib import Path
import argparse,ctypes,pickle,time,json,sys
import numpy as np
import scipy.sparse as sp
import polars as pl
from sparse_dot_topn import sp_matmul_topn
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;I=R/'experiments/astra_004'
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=2000);p.add_argument('--keep',type=int,default=8);p.add_argument('--k',type=int,default=100);p.add_argument('--build-only',action='store_true');args=p.parse_args();lib=ctypes.CDLL(str(O/'prune.dylib'));fun=lib.prune_columns;fun.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ctypes.c_int32,ctypes.c_int32,ctypes.c_int32];fun.restype=None
v={f:pickle.loads((R/f'experiments/astra_002/{f}_vectorizer.pkl').read_bytes()) for f in ['name','addr']};boundary=len(v['name'].vocabulary_);t=time.time()
for path in sorted(I.glob('train_s*.npz')):
 dest=O/f'keep{args.keep}_{path.name}'
 if dest.exists():continue
 m=sp.load_npz(path).tocsc();assert m.indices.dtype==m.indptr.dtype==np.int32
 before=m.nnz;fun(m.data,m.indices,m.indptr,m.shape[1],boundary,args.keep);m.eliminate_zeros();sp.save_npz(dest,m.tocsr(),compressed=False);print('pruned',path.name,before,m.nnz,round(time.time()-t),flush=True)
if args.build_only:sys.exit(0)
nt=pl.read_parquet(R/'experiments/astra_baseline/cache/ntrue_dev_20000_200.parquet').sort('s1').head(args.n);query=pl.read_parquet(R/'work/cache/train_source1.parquet').join(nt.select(pl.col('s1').alias('eid')),on='eid',how='semi').sort('eid');q=sp.hstack([v['name'].transform(query['name'].to_list())*.70710678,v['addr'].transform(query['addr'].to_list())*.70710678],format='csr',dtype=np.float32);best=None;covered=0;retrieval=time.time()
for src in [2,3]:
 path=R/f'work/cache/train_source{src}.parquet';total=pl.scan_parquet(path).select(pl.len()).collect().item()
 for start in range(0,total,250000):
  file=O/f'keep{args.keep}_train_s{src}_{start}.npz'
  if not file.exists():raise RuntimeError(f'Missing full-pool shard: {file}; rerun after full index build completes')
  mat=sp.load_npz(file);ids=pl.scan_parquet(path).slice(start,mat.shape[1]).select('eid').collect()['eid'].to_numpy();covered+=len(ids)
  z=sp_matmul_topn(q,mat,top_n=args.k,threshold=.01,sort=True,n_threads=2).tocoo();new=pl.DataFrame({'s1':query['eid'].to_numpy()[z.row],'src':np.full(z.nnz,src,dtype='uint8'),'eid':ids[z.col],'pruned_retrieval':z.data});best=new if best is None else pl.concat([best,new]);best=best.sort(['s1','pruned_retrieval','src','eid'],descending=[False,True,False,False]).group_by('s1',maintain_order=True).head(args.k);print('retrieved',src,start,round(time.time()-retrieval),flush=True)
best.write_parquet(O/f'keep{args.keep}_top{args.k}_{args.n}.parquet');truth=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(nt.select('s1'),on='s1',how='semi');base=pl.read_parquet(R/'experiments/astra_006/features.parquet').select('s1','src','eid');rows=[]
for k in [10,20,50,args.k]:
 c=best.group_by('s1',maintain_order=True).head(k).select('s1','src','eid');union=pl.concat([base,c]).unique();rows.append({'k':k,'keep_per_field':args.keep,'n_s1':query.height,'covered_records':covered,'retrieval_recall':truth.join(c,on=['s1','src','eid'],how='semi').height/truth.height,'union_recall':truth.join(union,on=['s1','src','eid'],how='semi').height/truth.height,'union_candidates':union.height,'retrieval_runtime':time.time()-retrieval})
pl.DataFrame(rows).write_csv(O/f'keep{args.keep}_ablation.csv');print(rows,flush=True)
