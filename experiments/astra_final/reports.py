"""Consolidate versioned measurements. Never invent submission or leaderboard status."""
from pathlib import Path
import csv,json,shutil
import polars as pl
R=Path(__file__).resolve().parents[2];O=R/'reports';rows=[]
def add(identifier,change,m,scope,decision,reason):
 rows.append({'experiment_id':identifier,'change':change,'validation_f05':m.get('validation_f05',m.get('macro_f05')),'precision':m.get('precision',m.get('macro_precision',m.get('validation_precision'))),'recall':m.get('recall',m.get('macro_recall',m.get('validation_recall'))),'singleton_accuracy':m.get('singleton_accuracy'),'candidate_recall':m.get('candidate_recall'),'runtime':m.get('runtime'),'decision':decision,'reason':reason,'evaluation_set':scope,'n_entities':m.get('n_entities'),'threshold':m.get('threshold')})
for name,scope in [('reports/astra_experiments.csv','development_20000'),('experiments/astra_003/experiments.csv','development_20000'),('experiments/astra_011/experiments.csv','development_20000'),('experiments/astra_013/experiments.csv','development_20000')]:
 p=R/name
 if p.exists():
  for m in pl.read_csv(p).to_dicts():add(m['experiment_id'],m.get('change',''),m,scope,m.get('decision',m.get('kept','measured')),m.get('reason','versioned comparison'))
for number in ['005','006','009','010']:
 p=R/f'experiments/astra_{number}/results.json'
 if p.exists():add('ASTRA'+number,'expanded candidate pilot',json.loads(p.read_text()),'development_pilot_2000','diagnostic','Pilot results are not full-validation or leaderboard scores')
for name in ['experiments/astra_015/ablation.csv','experiments/astra_015/candidate_floor.csv','experiments/astra_012/late_gate/ablation.csv']:
 for i,m in enumerate(pl.read_csv(R/name).to_dicts()):add(name+':'+str(i),'decision ablation',m,'development_20000','measured',json.dumps({k:m[k] for k in ['temperature','logit_shift','min_top_score','candidate_floor','gate_threshold'] if k in m}))
for name in ['experiments/astra_012/guard_v2_relative/guard_ablation.csv','experiments/astra_013/guard_macro/guard_ablation.csv']:
 for i,m in enumerate(pl.read_csv(R/name).to_dicts()):add(name+':'+str(i),'batch-invariant reference abstention',m,'development_20000','measured',f"strong={m['strong']}; margin={m['margin']}")
selected=json.loads((R/'experiments/astra_015/stability.json').read_text());add('ASTRA_SELECTED_DEVELOPMENT','expected F0.5 with singleton guard and individual score floor',selected['chosen'],'development_20000','selected','Frozen before reserved holdout; measured gain in both development halves')
hold=json.loads((R/'experiments/astra_final/holdout/metrics.json').read_text());add('ASTRA_FINAL_RESERVED_HOLDOUT','refit on 80000 S1; frozen pipeline',hold,'reserved_holdout_20000','confirmation','No holdout labels used for training, thresholds, or early stopping')
bp=R/'experiments/astra_final/baseline_holdout/metrics.json'
if bp.exists():add('ASTRA_BASELINE_SAME_HOLDOUT','protected original model and original blocker',json.loads(bp.read_text()),'reserved_holdout_20000','reference','Paired comparison on identical S1 entities')
with (O/'experiments.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name in ['dataset_forensics.md','blocking_ablation.csv']:
 p=O/name;backup=O/('pre_astra_'+name)
 if p.exists() and not backup.exists():shutil.copy2(p,backup)
shutil.copy2(O/'astra_dataset_forensics.md',O/'dataset_forensics.md')
blocks=[]
for file in ['experiments/astra_engine/blocking_ablation.csv','experiments/astra_engine/joint_ablation.csv','experiments/astra_004/dev_2000_ablation.csv','experiments/astra_008/keep8_ablation.csv']:
 p=R/file
 if p.exists():
  for i,m in enumerate(pl.read_csv(p).to_dicts()):blocks.append({'artifact':file,'variant':i,**m})
pl.from_dicts(blocks,infer_schema_length=None).write_csv(O/'blocking_ablation.csv')
print('Consolidated',len(rows),'experiments and',len(blocks),'blocking variants',flush=True)
