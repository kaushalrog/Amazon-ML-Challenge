"""Versioned, train-fitted evidence features. No evaluation labels used."""
import json,re,math
from pathlib import Path
import numpy as np
import polars as pl
from rapidfuzz import fuzz,process
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
EXTRA=['mapped_name_tsr','mapped_name_ratio','mapped_name_jaccard','name_idf_jaccard','name_idf_coverage','name_rare_shared','name_rare_missing','name_information','name_missing_either','house_number_equal','house_number_conflict','last_number_equal','last_number_conflict','numeric_coverage','name_consonant_ratio']
def resources():
 m=json.loads((O/'token_map.json').read_text());d=json.loads((O/'name_df.json').read_text());idf={k:math.log((1+d['n'])/(1+v)) for k,v in d['df'].items()};return m,idf,math.log(1+d['n'])
def compute(j,res):
 mapping,idf,unseen=res;n1=j['n1'].to_list();n2=j['n2'].to_list();mapped=[' '.join(mapping.get(t,t) for t in s.split()) for s in n2]
 rows=[]
 for a,b,raw,x,y in zip(n1,mapped,n2,j['a1'],j['a2']):
  aa=set(a.split());bb=set(b.split());shared=aa&bb;union=aa|bb;weights=lambda tokens:sum(idf.get(t,unseen) for t in tokens);sharedw=weights(shared);allw=weights(union);aw=weights(aa);nums1=re.findall(r'\d+',x);nums2=re.findall(r'\d+',y);both=bool(nums1 and nums2)
  rows.append((len(shared)/max(len(union),1),sharedw/max(allw,1e-6),sharedw/max(aw,1e-6),max((idf.get(t,unseen) for t in shared),default=0),max((idf.get(t,unseen) for t in aa-bb),default=0),max((idf.get(t,unseen) for t in aa),default=0),float(not a or not raw),float(both and nums1[0]==nums2[0]),float(both and nums1[0]!=nums2[0]),float(both and nums1[-1]==nums2[-1]),float(both and nums1[-1]!=nums2[-1]),len(set(nums1)&set(nums2))/max(len(set(nums1)),1)))
 ar=np.asarray(rows,dtype=np.float32)
 out=j.select('s1','src','eid')
 for i,col in enumerate(EXTRA[2:-1]):out=out.with_columns(pl.Series(col,ar[:,i]))
 out=out.with_columns(pl.Series('mapped_name_tsr',process.cpdist(n1,mapped,scorer=fuzz.token_sort_ratio,workers=2)/100),pl.Series('mapped_name_ratio',process.cpdist(n1,mapped,scorer=fuzz.ratio,workers=2)/100),pl.Series('name_consonant_ratio',process.cpdist([re.sub('[aeiou]','',s) for s in n1],[re.sub('[aeiou]','',s) for s in mapped],scorer=fuzz.ratio,workers=2)/100))
 return out
if __name__=='__main__':
 import argparse,time
 p=argparse.ArgumentParser();p.add_argument('fold');args=p.parse_args();fold=args.fold;n=60000 if fold=='train' else 20000;t=time.time();res=resources()
 x=pl.read_parquet(R/f'experiments/astra_baseline/cache/feat_{fold}_{n}_200.parquet').select('s1','src','eid')
 left=pl.scan_parquet(R/'work/cache/train_source1.parquet').rename({'eid':'s1'}).join(x.lazy().select('s1').unique(),on='s1',how='semi').select('s1',pl.col('name').alias('n1'),pl.col('addr').alias('a1')).collect(engine='streaming')
 right=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{s}.parquet').join(x.lazy().filter(pl.col('src')==s).select('eid').unique(),on='eid',how='semi').select('src','eid',pl.col('name').alias('n2'),pl.col('addr').alias('a2')).collect(engine='streaming') for s in [2,3]])
 out=[]
 for start in range(0,x.height,100000):
  j=x.slice(start,100000).join(left,on='s1').join(right,on=['src','eid']);out.append(compute(j,res));print(fold,min(start+100000,x.height),x.height,round(time.time()-t),flush=True)
 pl.concat(out).write_parquet(O/f'extra_{fold}.parquet');print('DONE',fold,time.time()-t,flush=True)
