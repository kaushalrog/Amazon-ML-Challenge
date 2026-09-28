"""Learn spelling/token correspondences strictly from model-training positives."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import polars as pl
from rapidfuzz.fuzz import ratio
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;C=R/'work/cache'
ids=pl.read_parquet(R/'experiments/astra_baseline/cache/ntrue_train_60000_200.parquet').select('s1')
s1=pl.read_parquet(C/'train_source1.parquet').rename({'eid':'s1'}).join(ids,on='s1',how='semi')
gt=pl.read_parquet(C/'gt_pairs.parquet').join(ids,on='s1',how='semi')
pool=pl.concat([pl.scan_parquet(C/f'train_source{s}.parquet').join(gt.lazy().filter(pl.col('src')==s).select('src','eid').unique(),on=['src','eid'],how='semi').collect() for s in [2,3]])
j=gt.join(s1.select('s1',pl.col('name').alias('n1')),on='s1').join(pool.select('src','eid',pl.col('name').alias('n2')),on=['src','eid'])
a=Counter();b=Counter();co=defaultdict(Counter)
for left,rights in j.group_by('s1').agg(pl.col('n1').first(),pl.col('n2')).select('n1','n2').iter_rows():
 lt=set(left.split());rt=set(' '.join(rights).split());a.update(lt);b.update(rt)
 for r in rt:co[r].update(lt)
# Symmetric association helps avoid mapping every rare spelling onto ubiquitous suffixes.
learned={};detail=[]
for r,counts in co.items():
 options=[]
 for l,c in counts.items():
  if c<3:continue
  assoc=2*c/(a[l]+b[r]);coverage=c/b[r];sim=ratio(r,l)/100
  if assoc>=.12 and coverage>=.65 and (sim>=.25 or assoc>=.5):options.append((assoc,coverage,c,l,sim))
 if not options:continue
 assoc,coverage,c,l,sim=max(options)
 if r!=l:
  learned[r]=l;detail.append({'token':r,'canonical':l,'association':assoc,'coverage':coverage,'support':c,'similarity':sim})
(O/'token_map.json').write_text(json.dumps(learned,ensure_ascii=False,indent=2));pl.DataFrame(detail).sort('support',descending=True).write_csv(O/'token_map_audit.csv')
# Frozen train-only document frequencies, distinct per document.
n=s1.height;df=Counter()
for text in s1['name']:df.update(set(text.split()))
(O/'name_df.json').write_text(json.dumps({'n':n,'df':dict(df)}));print('training positive pairs',j.height,'learned replacements',len(learned),flush=True)
print(pl.DataFrame(detail).sort('support',descending=True).head(30))
