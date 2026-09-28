"""Reusable disk-backed capped blocking indexes, built from supplied normalized records.

Indexes use a fixed Polars UInt64 hash and preserve the pool's integer ID mapping.
The candidate matcher, not the hash, makes identity decisions. Code/library/input
fingerprints are recorded so incompatible indexes are never silently reused.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import polars as pl
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/astra_baseline/src'))
from blocking_v2 import KEYS as BASE_KEYS
from blocking_rare import rarest_tokens
MAP_PATH=R/'experiments/astra_005/token_map.json'
OCR_PATH=R/'experiments/astra_005/ocr_map.json'
MAPPING=json.loads(MAP_PATH.read_text());OCR=json.loads(OCR_PATH.read_text())
def mapped():return pl.col('name').str.split(' ').list.eval(pl.element().replace(MAPPING)).list.join(' ').str.replace_many(list(OCR),list(OCR.values()))
def compact():return mapped().str.replace_all(' ','')
def consonants():return mapped().str.replace_all('[aeiou ]','')
def alpha_address():return pl.col('addr').str.replace_all(r'\d+','').str.replace_all(' +',' ').str.strip_chars()
EXTRA_KEYS={
 'mapped_name':lambda:compact(),
 'mapped_sorted':lambda:mapped().str.split(' ').list.sort().list.join(''),
 'consonant_name':lambda:consonants(),
 'name_prefix_number':lambda:compact().str.slice(0,7)+'|'+pl.col('addr').str.extract(r'(\d+)',1),
 'consonant_prefix_number':lambda:consonants().str.slice(0,6)+'|'+pl.col('addr').str.extract(r'(\d+)',1),
 'address_without_numbers':lambda:alpha_address(),
 'address_sorted_without_numbers':lambda:alpha_address().str.split(' ').list.sort().list.join(' '),
}
KEYS=BASE_KEYS|EXTRA_KEYS
SEED=20260927

def signature(split):
 files=[R/f'work/cache/{split}_source{s}.parquet' for s in [2,3]]
 return {'polars':pl.__version__,'seed':SEED,'inputs':{str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in files},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'token_map_sha256':hashlib.sha256(MAP_PATH.read_bytes()).hexdigest(),'ocr_map_sha256':hashlib.sha256(OCR_PATH.read_bytes()).hexdigest()}

def index_root(split):return R/f'experiments/astra_engine/index_{split}'

def save_index(root,name,keys,positions):
 keys=np.concatenate(keys);positions=np.concatenate(positions);order=np.argsort(keys,kind='stable');np.save(root/f'{name}_keys.npy',keys[order]);np.save(root/f'{name}_positions.npy',positions[order])

def build(split,include_rare=True):
 root=index_root(split);root.mkdir(parents=True,exist_ok=True);meta=root/'manifest.json';sig=signature(split)
 if meta.exists():assert json.loads(meta.read_text())==sig,'Index fingerprint mismatch: use a new version directory'
 else:meta.write_text(json.dumps(sig,indent=2))
 paths=[R/f'work/cache/{split}_source{s}.parquet' for s in [2,3]];counts=[pl.scan_parquet(p).select(pl.len()).collect().item() for p in paths]
 if not (root/'eids.npy').exists():
  np.save(root/'eids.npy',np.concatenate([pl.read_parquet(p,columns=['eid'])['eid'].to_numpy() for p in paths]));np.save(root/'sources.npy',np.concatenate([np.full(n,s,dtype='uint8') for s,n in zip([2,3],counts)]))
 for name,fn in KEYS.items():
  if (root/f'{name}_positions.npy').exists():continue
  hashes=[];positions=[];offset=0;t=time.time()
  for path,n in zip(paths,counts):
   for start in range(0,n,250000):
    part=pl.scan_parquet(path).slice(start,250000).collect().with_row_index('_row',offset=offset+start).select('_row',fn().alias('key')).filter(pl.col('key').is_not_null()&(pl.col('key').str.len_chars()>=4))
    hashes.append(part['key'].hash(seed=SEED).to_numpy());positions.append(part['_row'].to_numpy())
   offset+=n
  save_index(root,name,hashes,positions);print(split,name,round(time.time()-t),flush=True)
 if include_rare:
  for col in ['name','addr']:
   name='rare_'+col
   if (root/f'{name}_positions.npy').exists():continue
   freqfile=root/f'{col}_frequency.parquet'
   if freqfile.exists():freq=pl.read_parquet(freqfile)
   else:
    freq=pl.concat([pl.scan_parquet(p).select(pl.col(col).str.split(' ').alias('t')).explode('t').filter(pl.col('t').str.len_chars()>=3).group_by('t').len().collect(engine='streaming') for p in paths]).group_by('t').agg(pl.col('len').sum().alias('df'));freq.write_parquet(freqfile)
   hashes=[];positions=[];offset=0;t=time.time()
   for path,n in zip(paths,counts):
    for start in range(0,n,250000):
     part=pl.scan_parquet(path).slice(start,250000).collect().with_row_index('_row',offset=offset+start)
     tok=rarest_tokens(part,freq,k=2,col=col,id_cols=('_row',));hashes.append(tok['t'].hash(seed=SEED).to_numpy());positions.append(tok['_row'].to_numpy())
    offset+=n
   save_index(root,name,hashes,positions);print(split,name,round(time.time()-t),flush=True)

class Index:
 def __init__(self,split):
  self.root=index_root(split);assert json.loads((self.root/'manifest.json').read_text())==signature(split),'Index fingerprint mismatch'
  self.ids=np.load(self.root/'eids.npy',mmap_mode='r');self.src=np.load(self.root/'sources.npy',mmap_mode='r');self.keys={};self.positions={}
  for p in self.root.glob('*_keys.npy'):
   name=p.name[:-9];self.keys[name]=np.load(p,mmap_mode='r');self.positions[name]=np.load(self.root/f'{name}_positions.npy',mmap_mode='r')
  self.freq={col:pl.read_parquet(self.root/f'{col}_frequency.parquet') for col in ['name','addr'] if (self.root/f'{col}_frequency.parquet').exists()}
 def query_key(self,ids,strings,name,cap):
  hashes=strings.hash(seed=SEED).to_numpy();keys=self.keys[name];lo=np.searchsorted(keys,hashes,side='left');hi=np.searchsorted(keys,hashes,side='right');counts=hi-lo;keep=(counts>0)&(counts<=cap);lo=lo[keep];counts=counts[keep];ids=ids.to_numpy()[keep]
  if not len(ids):return pl.DataFrame(schema={'s1':pl.Int64,'src':pl.UInt8,'eid':pl.Int64})
  end=np.cumsum(counts);ii=np.repeat(lo,counts)+np.arange(end[-1])-np.repeat(end-counts,counts);pos=self.positions[name][ii]
  return pl.DataFrame({'s1':np.repeat(ids,counts),'src':self.src[pos],'eid':self.ids[pos]})
 def candidates(self,s1,cap=200,extra=False):
  parts=[]
  for name,fn in (KEYS if extra else BASE_KEYS).items():
   q=s1.select('eid',fn().alias('key')).filter(pl.col('key').is_not_null()&(pl.col('key').str.len_chars()>=4));parts.append(self.query_key(q['eid'],q['key'],name,cap))
  for col,freq in self.freq.items():
   q=rarest_tokens(s1,freq,k=2,col=col,id_cols=('eid',));parts.append(self.query_key(q['eid'],q['t'],'rare_'+col,cap))
  return pl.concat(parts).unique()
if __name__=='__main__':build(sys.argv[1])
