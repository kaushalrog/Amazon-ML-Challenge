"""Joint name/address retrieval; reusable disk shards and frozen training vectorizers."""
import argparse,pickle,json,time,sys
from pathlib import Path
import polars as pl
import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;O.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=2000);p.add_argument('--k',type=int,default=100);p.add_argument('--split',default='dev');p.add_argument('--shard',type=int,default=250000);args=p.parse_args()
vs={f:pickle.loads((R/f'experiments/astra_002/{f}_vectorizer.pkl').read_bytes()) for f in ['name','addr']};mapping=json.loads((R/'experiments/astra_003/token_map.json').read_text())
def matrix(df,is_pool=False):
 names=df['name'].to_list()
 if is_pool:names=[' '.join(mapping.get(t,t) for t in s.split()) for s in names]
 return sp.hstack([vs['name'].transform(names)*np.float32(.70710678),vs['addr'].transform(df['addr'].to_list())*np.float32(.70710678)],format='csr')
if args.split=='test':query=pl.read_parquet(R/'work/cache/test_source1.parquet').head(args.n);data='test'
else:
 n=60000 if args.split=='train' else 20000;ids=pl.read_parquet(R/f'experiments/astra_baseline/cache/ntrue_{args.split}_{n}_200.parquet').select(pl.col('s1').alias('eid')).sort('eid').head(args.n);query=pl.read_parquet(R/'work/cache/train_source1.parquet').join(ids,on='eid',how='semi').sort('eid');data='train'
q=matrix(query);best=None;t=time.time();print('query',query.height,q.shape,flush=True)
for src in [2,3]:
 path=R/f'work/cache/{data}_source{src}.parquet';total=pl.scan_parquet(path).select(pl.len()).collect().item()
 for start in range(0,total,args.shard):
  stem=O/f'{data}_s{src}_{start}';part=pl.scan_parquet(path).slice(start,args.shard).select('eid','name','addr').collect()
  if stem.with_suffix('.npz').exists():mat=sp.load_npz(stem.with_suffix('.npz'))
  else:mat=matrix(part,True).T.tocsr();sp.save_npz(stem.with_suffix('.npz'),mat,compressed=False)
  z=sp_matmul_topn(q,mat,top_n=args.k,threshold=.10,sort=True,n_threads=2).tocoo();new=pl.DataFrame({'s1':query['eid'].to_numpy()[z.row],'src':np.full(z.nnz,src,dtype='uint8'),'eid':part['eid'].to_numpy()[z.col],'joint_retrieval':z.data});best=new if best is None else pl.concat([best,new]);best=best.sort(['s1','joint_retrieval','src','eid'],descending=[False,True,False,False]).group_by('s1',maintain_order=True).head(args.k);print(src,start+part.height,round(time.time()-t),flush=True)
best.write_parquet(O/f'{args.split}_top{args.k}_{args.n}.parquet')
if args.split!='test':
 truth=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(query.select(pl.col('eid').alias('s1')),on='s1',how='semi');baseline=pl.read_parquet(R/f'experiments/astra_baseline/cache/feat_{args.split}_{60000 if args.split=="train" else 20000}_200.parquet').select('s1','src','eid').join(query.select(pl.col('eid').alias('s1')),on='s1',how='semi');rows=[]
 for k in [5,10,20,50,args.k]:
  b=best.group_by('s1',maintain_order=True).head(k).select('s1','src','eid');u=pl.concat([baseline,b]).unique();rows.append({'k':k,'n_s1':query.height,'retrieval_recall':truth.join(b,on=['s1','src','eid'],how='semi').height/truth.height,'union_recall':truth.join(u,on=['s1','src','eid'],how='semi').height/truth.height,'union_pairs':u.height,'runtime':time.time()-t})
 pl.DataFrame(rows).write_csv(O/f'{args.split}_{args.n}_ablation.csv');print(rows,flush=True)
