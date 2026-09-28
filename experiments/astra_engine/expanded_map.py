"""Additional train-only spelling evidence fitted on 350,000 distinct S1 groups."""
from pathlib import Path
import json
import polars as pl
from rapidfuzz import process,fuzz
R=Path(__file__).resolve().parents[2]
FEATURES=['expanded_name_tsr','expanded_name_ratio','expanded_name_jaccard']
def resources():return json.loads((R/'experiments/astra_013/token_map.json').read_text())
def compute(j,mapping):
 z=j.with_columns(pl.col('n2').str.split(' ').list.eval(pl.element().replace(mapping)).list.join(' ').alias('mapped'))
 a=z['n1'].to_list();b=z['mapped'].to_list();aa=pl.col('n1').str.split(' ');bb=pl.col('mapped').str.split(' ')
 return z.select('s1','src','eid',(aa.list.set_intersection(bb).list.len()/aa.list.set_union(bb).list.len().clip(1)).cast(pl.Float32).alias(FEATURES[2])).with_columns(pl.Series(FEATURES[0],process.cpdist(a,b,scorer=fuzz.token_sort_ratio,workers=2)/100),pl.Series(FEATURES[1],process.cpdist(a,b,scorer=fuzz.ratio,workers=2)/100))
