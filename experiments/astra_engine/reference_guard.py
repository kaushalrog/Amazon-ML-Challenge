"""Conservative abstention when supplied reference text supports another S1 more strongly.

This never assigns a candidate to another entity, and never limits S1 match count.
No alternative S1 ground-truth labels enter this decision.
"""
from pathlib import Path
import sys
import numpy as np,polars as pl,lightgbm as lgb
R=Path(__file__).resolve().parents[2];B=R/'experiments/astra_baseline';sys.path.insert(0,str(B/'src'));import features_v2 as F
KEY=['s1','src','eid'];SEED=20260927
class ReferenceGuard:
 def __init__(self,s1):
  self.s1=s1.sort('eid');self.ids=self.s1['eid'].to_numpy();self.index={};self.model=lgb.Booster(model_file=str(B/'models/matcher_lgbm.txt'))
  for col in ['name','addr']:
   z=s1.filter(pl.col(col)!='').select('eid',pl.col(col).hash(seed=SEED).alias('hash')).sort('hash');self.index[col]=(z['hash'].to_numpy(),z['eid'].to_numpy())
 def enrich(self,scored,pool):
  eligible=scored.filter(pl.col('score')>=.02).select(KEY)
  if not eligible.height:return scored.with_columns(pl.lit(None,dtype=pl.Float32).alias('fixed_count_score'),pl.lit(0.,dtype=pl.Float32).alias('alternative_score'))
  targets=pool.join(eligible.select('src','eid').unique(),on=['src','eid'],how='semi');parts=[eligible]
  for field,(hh,ii) in self.index.items():
   hs=targets[field].hash(seed=SEED).to_numpy();lo=np.searchsorted(hh,hs,'left');hi=np.searchsorted(hh,hs,'right');nn=hi-lo;valid=(nn>0)&(nn<=100)&(targets[field].str.len_chars().to_numpy()>0);lo=lo[valid];nn=nn[valid]
   if not len(nn):continue
   end=np.cumsum(nn);pos=np.repeat(lo,nn)+np.arange(end[-1])-np.repeat(end-nn,nn);parts.append(pl.DataFrame({'s1':ii[pos],'src':np.repeat(targets['src'].to_numpy()[valid],nn),'eid':np.repeat(targets['eid'].to_numpy()[valid],nn)}))
  altpairs=pl.concat(parts[1:]).unique() if len(parts)>1 else eligible.head(0);pairs=pl.concat(parts).unique();sids=pairs['s1'].unique().to_numpy();pos=np.searchsorted(self.ids,sids);assert np.array_equal(self.ids[pos],sids);j=F.attach_records(pairs,self.s1[pos],targets);x=F.compute(j).with_columns(pl.lit(100.,dtype=pl.Float32).alias('cand_count'));fixed=x.select(KEY).with_columns(pl.Series('fixed_count_score',self.model.predict(x.select(F.FEATURES).to_numpy(),num_threads=2).astype('float32')))
  best=fixed.join(altpairs,on=KEY,how='semi').sort('fixed_count_score',descending=True).group_by('src','eid',maintain_order=True).agg(pl.col('s1').first().alias('best_s1'),pl.col('fixed_count_score').first().alias('best'),pl.col('fixed_count_score').slice(1,1).first().alias('second'))
  current=eligible.join(fixed,on=KEY).join(best,on=['src','eid'],how='left').with_columns(pl.when(pl.col('s1')!=pl.col('best_s1')).then(pl.col('best')).otherwise(pl.col('second')).fill_null(0.).alias('alternative_score')).select(*KEY,'fixed_count_score','alternative_score')
  return scored.join(current,on=KEY,how='left').with_columns(pl.col('alternative_score').fill_null(0.))
def accepted(scored,config):
 threshold=pl.col('src').cast(pl.String).replace_strict(config['source_thresholds'],return_dtype=pl.Float32) if 'source_thresholds' in config else pl.lit(config['threshold'])
 good=pl.lit(True) if 'probabilistic_decision' in config else pl.col('score')>=threshold
 if 'full_feature_gate' in config:good=good&(pl.col('gate_score')>=config['full_feature_gate']['threshold'])
 if 'reference_guard' in config:
  g=config['reference_guard'];reject=(pl.col('alternative_score')>=g['strong'])&(pl.col('alternative_score')>pl.col('fixed_count_score')+g['margin']);good=good&~reject.fill_null(False)
 filtered=scored.filter(good)
 if 'probabilistic_decision' in config:
  from probabilistic_decision import choose
  return choose(filtered,config)
 return filtered
