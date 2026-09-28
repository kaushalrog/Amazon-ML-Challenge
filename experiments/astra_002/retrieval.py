"""Train-fitted character TF-IDF retrieval over the complete source pool, in bounded shards."""
import os,sys,time,json,pickle,argparse
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'experiments/astra_baseline/src'))
from split import load_fold
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=2000);p.add_argument('--k',type=int,default=20);p.add_argument('--chunk',type=int,default=250000);args=p.parse_args()
# Avoid snapshot module's relocated ROOT; load fold IDs directly.
base=pl.read_parquet(R/'work/cache/train_source1.parquet');trainids=pl.read_parquet(R/'experiments/astra_baseline/cache/ntrue_train_60000_200.parquet').rename({'s1':'eid'})
dev_ids=pl.read_parquet(R/'experiments/astra_baseline/cache/ntrue_dev_20000_200.parquet').sort('s1').head(args.n).rename({'s1':'eid'})
train=base.join(trainids.select('eid'),on='eid',how='semi');query=base.join(dev_ids.select('eid'),on='eid',how='semi').sort('eid');del base
truth=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(query.select(pl.col('eid').alias('s1')),on='s1',how='semi');basepairs=pl.read_parquet(R/'experiments/astra_baseline/cache/dev_scored.parquet').join(query.select(pl.col('eid').alias('s1')),on='s1',how='semi').select('s1','src','eid')
rows=[];allnew=[];t0=time.time()
for field in ['name','addr']:
 path=O/f'{field}_vectorizer.pkl'
 if path.exists():v=pickle.loads(path.read_bytes())
 else:
  v=TfidfVectorizer(analyzer='char',ngram_range=(3,4),min_df=2,max_df=.01,dtype=np.float32,sublinear_tf=True);v.fit(train[field].to_list());path.write_bytes(pickle.dumps(v))
 qmat=v.transform(query[field].to_list());best=None;tf=time.time();print(field,'vocab',len(v.vocabulary_),'query',qmat.shape,flush=True)
 for src in [2,3]:
  path=R/f'work/cache/train_source{src}.parquet';total=pl.scan_parquet(path).select(pl.len()).collect().item()
  for start in range(0,total,args.chunk):
   part=pl.scan_parquet(path).slice(start,args.chunk).select('eid',field).collect();mat=v.transform(part[field].to_list())
   scores=sp_matmul_topn(qmat,mat.T.tocsr(),top_n=args.k,threshold=.10,sort=True,n_threads=2).tocoo()
   new=pl.DataFrame({'s1':query['eid'].to_numpy()[scores.row],'src':np.full(scores.nnz,src,dtype='uint8'),'eid':part['eid'].to_numpy()[scores.col],f'{field}_retrieval':scores.data})
   best=new if best is None else pl.concat([best,new]);best=best.sort(['s1',f'{field}_retrieval','src','eid'],descending=[False,True,False,False]).group_by('s1',maintain_order=True).head(args.k)
   print(field,src,start+part.height,'seconds',round(time.time()-tf),flush=True)
 best.write_parquet(O/f'{field}_top{args.k}_{args.n}.parquet');allnew.append(best.select('s1','src','eid'))
 for k in [5,10,args.k]:
  cand=best.group_by('s1',maintain_order=True).head(k).select('s1','src','eid');union=pl.concat([basepairs,cand]).unique();rows.append({'field':field,'k':k,'n_s1':query.height,'candidates':cand.height,'candidate_recall':truth.join(cand,on=['s1','src','eid'],how='semi').height/truth.height,'union_candidates':union.height,'union_recall':truth.join(union,on=['s1','src','eid'],how='semi').height/truth.height,'runtime':time.time()-tf})
 pl.DataFrame(rows).write_csv(O/'blocking_ablation.csv')
union=pl.concat([basepairs,*allnew]).unique();union.write_parquet(O/f'candidate_union_{args.n}.parquet');rows.append({'field':'UNION','k':args.k,'n_s1':query.height,'candidates':union.height,'candidate_recall':truth.join(union,on=['s1','src','eid'],how='semi').height/truth.height,'union_candidates':union.height,'union_recall':truth.join(union,on=['s1','src','eid'],how='semi').height/truth.height,'runtime':time.time()-t0});pl.DataFrame(rows).write_csv(O/'blocking_ablation.csv');print(rows,flush=True)
