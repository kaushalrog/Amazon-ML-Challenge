"""Dev-only expected-macro-F0.5 prefix decisions, with calibrated probability sensitivity."""
from pathlib import Path
import sys,json,ctypes,time
import numpy as np,polars as pl
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R/'experiments/astra_baseline/src'));import evaluate_v2 as E
s=pl.read_parquet(R/'experiments/astra_013/guard_macro/scored.parquet').join(pl.read_parquet(R/'experiments/astra_final/dev_gate_scores.parquet'),on=['s1','src','eid']);s=s.filter((pl.col('gate')>=.005)&~((pl.col('alternative_score')>=.9)&(pl.col('alternative_score')>pl.col('fixed_count_score')+.01)).fill_null(False));nt=pl.read_parquet(R/'experiments/astra_011/dev/ntrue.parquet').sort('s1');q=nt['s1'].to_numpy();K=20;s=s.sort(['s1','score'],descending=[False,True]).group_by('s1',maintain_order=True).head(K).with_columns((pl.col('s1').cum_count().over('s1')-1).alias('rank'));pos=np.searchsorted(q,s['s1'].to_numpy());rank=s['rank'].to_numpy();a=np.zeros((len(q),K),dtype='float32');a[pos,rank]=s['score'].to_numpy();lib=ctypes.CDLL(str(R/'experiments/astra_engine/expected_f05.dylib'));f=lib.expected_f05;f.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')];rows=[];t=time.time()
for temperature in [.7,1.,1.3]:
 for shift in [-1.,-.5,0.,.5]:
  p=1/(1+np.exp(-(np.log(np.clip(a,1e-7,1-1e-7)/(1-np.clip(a,1e-7,1-1e-7)))/temperature+shift)));p[a==0]=0;kk=np.empty(len(q),dtype='int32');f(p.astype('float32'),len(q),K,kk);
  for top_guard in [0.,.7,.75,.775,.8,.85,.9]:
   counts=kk.copy();counts[a.max(axis=1)<top_guard]=0;sel=s.filter(pl.Series(rank<counts[pos]));m=E.evaluate(sel,nt,0.);m.update(temperature=temperature,logit_shift=shift,min_top_score=top_guard,runtime=time.time()-t);rows.append(m)
pl.DataFrame(rows).sort('macro_f05',descending=True).write_csv(O/'ablation.csv');print(pl.DataFrame(rows).sort('macro_f05',descending=True).head(5).to_dicts(),flush=True)
