"""Compare original and bounded-memory validator on malformed and valid fixtures."""
import importlib.util,tempfile,contextlib,io,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load(R/'experiments/astra_baseline/utils/validate_submission.py','old');new=load(R/'utils/validate_submission.py','new')
base='source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n';cand='source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n';cases={'valid':(base,cand),'outside_candidate':(base,cand.replace('S2-1,S3-1','S3-1')),'duplicate_candidate':(base,cand.replace('S2-1,S3-1','S2-1,S2-1')),'unknown':(base.replace('S2-1','S2-9'),cand.replace('S2-1','S2-9')),'missing':(base,cand.split('S1-2')[0]),'self':(base.replace('S2-1','S1-2'),cand),'malformed':(base,cand.replace('S1-1\t','S1-1 ')),'duplicate_row':(base,cand+'S1-1\tS2-1\n')};rows=[]
with tempfile.TemporaryDirectory() as td:
 p=Path(td)
 for s,ids in [(1,['S1-1','S1-2']),(2,['S2-1']),(3,['S3-1'])]:(p/f'test_source{s}.tsv').write_text('entity_id\n'+'\n'.join(ids)+'\n')
 for name,(m,c) in cases.items():
  (p/'m.tsv').write_text(m);(p/'c.tsv').write_text(c)
  with contextlib.redirect_stdout(io.StringIO()):a=old.validate(str(p/'m.tsv'),str(p/'c.tsv'),str(p),True);b=new.validate(str(p/'m.tsv'),str(p/'c.tsv'),str(p),True)
  assert a[0]==b[0],(name,a,b)
  if name not in ['missing','malformed']:assert a[1]==b[1],(name,a,b)
  rows.append({'case':name,'same_failures':True,'original_errors':len(a[0]),'new_errors':len(b[0])})
(R/'reports/astra_validator_regression.json').write_text(json.dumps(rows,indent=2));print('PASS',len(rows),'fixtures; identical failures')
