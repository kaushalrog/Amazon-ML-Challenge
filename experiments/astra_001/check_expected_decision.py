from pathlib import Path
import sys,ctypes,itertools,json
import numpy as np,polars as pl
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/astra_engine'));from reference_guard import accepted
lib=ctypes.CDLL(str(R/'experiments/astra_engine/expected_f05.dylib'));f=lib.expected_f05;f.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')];rng=np.random.default_rng(17)
for n in range(1,6):
 for _ in range(20):
  p=np.sort(rng.uniform(0,1,n).astype('float32'))[::-1].copy();utility=np.zeros(n+1)
  for bits in itertools.product([0,1],repeat=n):
   y=np.array(bits);pr=np.prod(np.where(y,p,1-p).astype('float64'));utility[0]+=pr*(sum(y)==0)
   for k in range(1,n+1):utility[k]+=pr*(5*sum(y[:k])/(4*k+sum(y)))
  out=np.empty(1,dtype='int32');f(p,1,n,out);assert out[0]==np.argmax(utility),(p,out,utility)
cfg=json.loads((R/'experiments/astra_013/selected_decision.json').read_text());cfg['full_feature_gate']={'threshold':.005};cfg['probabilistic_decision']={'temperature':1.3,'logit_shift':.5,'min_top_score':.775,'candidate_floor':.5,'max_candidates':20};s=pl.read_parquet(R/'experiments/astra_013/guard_macro/scored.parquet').join(pl.read_parquet(R/'experiments/astra_final/dev_gate_scores.parquet').rename({'gate':'gate_score'}),on=['s1','src','eid']);out=accepted(s,cfg);expected=pl.read_parquet(R/'experiments/astra_015/selected.parquet');key=['s1','src','eid'];assert out.select(key).sort(key).equals(expected.select(key).sort(key));(R/'reports/astra_expected_decision_checks.json').write_text(json.dumps({'brute_force_cases':100,'development_selected_pairs':out.height,'exact_policy_reproduction':True},indent=2));print('PASS: 100 exact brute-force cases and development prediction equality')
