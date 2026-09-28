"""Choose the ranked prefix maximizing expected per-entity F0.5, with abstention.

Scores are transformed using development-selected temperature/offset. Independent
Bernoulli probabilities are a decision approximation, not a claim of calibrated
certainty. A per-candidate floor prevents acceptance of weak individual scores.
"""
from pathlib import Path
import ctypes
import numpy as np,polars as pl
LIB=None

def choose(scored,config):
 global LIB
 if not scored.height:return scored
 cfg=config['probabilistic_decision'];K=cfg['max_candidates'];assert 1<=K<=32
 if LIB is None:
  LIB=ctypes.CDLL(str(Path(__file__).with_name('expected_f05.dylib')));LIB.expected_f05.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')]
 s=scored.filter(pl.col('score')>0).sort(['s1','score','src','eid'],descending=[False,True,False,False]).group_by('s1',maintain_order=True).head(K)
 if not s.height:return scored.head(0)
 s=s.with_columns((pl.col('s1').cum_count().over('s1')-1).alias('_rank'));ids=s['s1'].unique().sort().to_numpy();pos=np.searchsorted(ids,s['s1'].to_numpy());rank=s['_rank'].to_numpy();raw=np.zeros((len(ids),K),dtype='float32');raw[pos,rank]=s['score'].to_numpy();clip=np.clip(raw,1e-7,1-1e-7);pp=1/(1+np.exp(-(np.log(clip/(1-clip))/cfg['temperature']+cfg['logit_shift'])));pp[raw==0]=0;counts=np.empty(len(ids),dtype='int32');LIB.expected_f05(pp.astype('float32'),len(ids),K,counts);counts[raw.max(axis=1)<cfg['min_top_score']]=0
 return s.filter(pl.Series(rank<counts[pos])&(pl.col('score')>=cfg['candidate_floor'])).drop('_rank')
