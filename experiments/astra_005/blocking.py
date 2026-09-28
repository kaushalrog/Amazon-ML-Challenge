"""Targeted data-derived OCR, spelling and cross-field blocking ablation."""
import json,time,difflib
from collections import Counter
from pathlib import Path
import polars as pl
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;C=R/'work/cache';n=2000;cap=200;t=time.time()
learned=json.loads((R/'experiments/astra_003/token_map.json').read_text());df=json.loads((R/'experiments/astra_003/name_df.json').read_text())['df']
# Do not turn co-occurring English words into synonyms (e.g. ear/throat).
mapping={a:b for a,b in learned.items() if df.get(a,0)<3 and difflib.SequenceMatcher(None,a,b).ratio()>=.25};(O/'token_map.json').write_text(json.dumps(mapping,indent=2))
rules=Counter()
for a,b in mapping.items():
 for op,i,j,k,l in difflib.SequenceMatcher(None,a,b).get_opcodes():
  if op!='equal' and a[i:j].isdigit() and len(a[i:j])==len(b[k:l])==1 and b[k:l].isalpha():rules[(a[i:j],b[k:l])]+=1
ocr={a:b for (a,b),count in rules.items() if count>=10};(O/'ocr_map.json').write_text(json.dumps(ocr,indent=2))
def prepare(q):
 return q.with_columns(pl.col('name').str.split(' ').list.eval(pl.element().replace(mapping)).list.join(' ').str.replace_many(list(ocr),list(ocr.values())).alias('mapped'))
def expressions():
 compact=pl.col('mapped').str.replace_all(' ','');cons=pl.col('mapped').str.replace_all('[aeiou ]','');addr=pl.col('addr').str.replace_all(r'\d+','').str.replace_all(' +',' ').str.strip_chars()
 return {'mapped_name':compact,'mapped_sorted':pl.col('mapped').str.split(' ').list.sort().list.join(''),'consonant_name':cons,'name_prefix_number':compact.str.slice(0,7)+'|'+pl.col('addr').str.extract(r'(\d+)',1),'consonant_prefix_number':cons.str.slice(0,6)+'|'+pl.col('addr').str.extract(r'(\d+)',1),'address_without_numbers':addr,'address_sorted_without_numbers':addr.str.split(' ').list.sort().list.join(' ')}
ids=pl.read_parquet(R/'experiments/astra_baseline/cache/ntrue_dev_20000_200.parquet').sort('s1').head(n).select(pl.col('s1').alias('eid'));s1=prepare(pl.read_parquet(C/'train_source1.parquet').join(ids,on='eid',how='semi'));truth=pl.read_parquet(C/'gt_pairs.parquet').join(ids.rename({'eid':'s1'}),on='s1',how='semi');base=pl.read_parquet(R/'experiments/astra_baseline/cache/dev_scored.parquet').select('s1','src','eid').join(ids.rename({'eid':'s1'}),on='s1',how='semi');parts=[];rows=[]
for name,expr in expressions().items():
 tt=time.time();left=s1.select(pl.col('eid').alias('s1'),expr.alias('key')).filter(pl.col('key').str.len_chars()>=4);targets=left.select('key').unique();right=[]
 for src in [2,3]:
  q=prepare(pl.scan_parquet(C/f'train_source{src}.parquet')).select('src','eid',expr.alias('key')).join(targets.lazy(),on='key',how='semi');right.append(q.collect(engine='streaming'))
 right=pl.concat(right);counts=right.group_by('key').len();right=right.join(counts.filter(pl.col('len')<=cap).select('key'),on='key',how='semi');cand=left.join(right,on='key').select('s1','src','eid').unique();parts.append(cand);union=pl.concat([base,*parts]).unique();row={'rule':name,'candidates':cand.height,'recall':truth.join(cand,on=['s1','src','eid'],how='semi').height/truth.height,'union_candidates':union.height,'union_recall':truth.join(union,on=['s1','src','eid'],how='semi').height/truth.height,'runtime':time.time()-tt};rows.append(row);print(row,flush=True);pl.DataFrame(rows).write_csv(O/'ablation.csv');cand.write_parquet(O/f'{name}.parquet')
print('total seconds',time.time()-t,flush=True)
