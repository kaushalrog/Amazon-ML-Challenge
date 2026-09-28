"""Exact original blocking lookup with small directories to reduce random mmap reads."""
from pathlib import Path
import ctypes
import numpy as np,polars as pl
from key_index import Index as OriginalIndex,SEED
class Index(OriginalIndex):
 def __init__(self,split):
  super().__init__(split);lib=ctypes.CDLL(str(Path(__file__).with_name('bucket_lookup.dylib')));self.lookup=lib.bucket_lookup;self.lookup.argtypes=[np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS'),ctypes.c_int64,np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS')];self.directory={};bounds=np.arange(65536,dtype='uint64')<<np.uint64(48)
  for name,keys in self.keys.items():
   p=self.root/f'{name}_directory16.npy'
   if not p.exists():
    directory=np.concatenate([np.searchsorted(keys,bounds,'left'),[len(keys)]]).astype('int64');tmp=p.with_suffix('.tmp.npy');np.save(tmp,directory);tmp.replace(p)
   self.directory[name]=np.load(p);assert len(self.directory[name])==65537 and self.directory[name][-1]==len(keys)
 def query_key(self,ids,strings,name,cap):
  hashes=strings.hash(seed=SEED).to_numpy();lo=np.empty(len(hashes),dtype='int64');hi=np.empty_like(lo);self.lookup(self.keys[name],hashes,self.directory[name],len(hashes),lo,hi);counts=hi-lo;keep=(counts>0)&(counts<=cap);lo=lo[keep];counts=counts[keep];ids=ids.to_numpy()[keep]
  if not len(ids):return pl.DataFrame(schema={'s1':pl.Int64,'src':pl.UInt8,'eid':pl.Int64})
  end=np.cumsum(counts);ii=np.repeat(lo,counts)+np.arange(end[-1])-np.repeat(end-counts,counts);pos=self.positions[name][ii];return pl.DataFrame({'s1':np.repeat(ids,counts),'src':self.src[pos],'eid':self.ids[pos]})
if __name__=='__main__':
 import sys,time,json
 R=Path(__file__).resolve().parents[2];t=time.time();new=Index('test');old=OriginalIndex('test');q=pl.read_parquet(R/'work/cache/test_source1.parquet');rows=[]
 for start in [0,500000,1000000,1500000]:
  z=q.slice(start,500);tt=time.time();a=old.candidates(z,400,True);oldtime=time.time()-tt;tt=time.time();b=new.candidates(z,400,True);newtime=time.time()-tt;assert a.sort('s1','src','eid').equals(b.sort('s1','src','eid'));rows.append({'start':start,'pairs':a.height,'original_seconds':oldtime,'directory_seconds':newtime})
 (R/'reports/astra_index_equivalence.json').write_text(json.dumps({'batches':rows,'elapsed':time.time()-t,'exact_candidate_equality':True},indent=2));print(rows,flush=True)
