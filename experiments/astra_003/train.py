"""Ablate evidence features on the fixed real candidate sets; dev-only selection."""
from pathlib import Path
import sys,time,json
import polars as pl
import lightgbm as lgb
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;B=R/'experiments/astra_baseline'
sys.path.insert(0,str(B/'src'));import evaluate_v2 as E
basefeatures=json.loads((B/'models/decision.json').read_text())['features'];frames={};nts={}
if (O/'mined_train.parquet').exists():
 frames['train']=pl.read_parquet(O/'mined_train.parquet');weights=frames['train']['_weight'].to_numpy();frames['train']=frames['train'].drop('_weight')
else:
 base_model=lgb.Booster(model_file=str(B/'models/matcher_lgbm.txt'))
 selected=[];allweights=[];rng=np.random.default_rng(20260927)
 path=B/'cache/feat_train_60000_200.parquet'
 total=pl.scan_parquet(path).select(pl.len()).collect().item()
 for start in range(0,total,150000):
  chunk=pl.scan_parquet(path).slice(start,150000).collect()
  p=base_model.predict(chunk.select(basefeatures).to_numpy(),num_threads=2)
  y=chunk['label'].to_numpy();hard=(y==1)|(p>=.05);keep=hard|(rng.random(len(y))<.05)
  selected.append(chunk.filter(pl.Series(keep)));allweights.append(np.where(hard[keep],1.,20.).astype('float32'))
  print('mine',start+chunk.height,total,flush=True)
 small=pl.concat(selected);weights=np.concatenate(allweights);del selected,allweights,chunk,p,y
 frames['train']=small.join(pl.scan_parquet(O/'extra_train.parquet').join(small.lazy().select('s1','src','eid'),on=['s1','src','eid'],how='semi').collect(engine='streaming'),on=['s1','src','eid'],validate='1:1',maintain_order='left');del small
 frames['train'].with_columns(pl.Series('_weight',weights)).write_parquet(O/'mined_train.parquet')
# Explicit row key checks preserve sample-weight alignment through joins.
# Polars left join preserves left order when maintain_order='left'.
frames['dev']=pl.read_parquet(B/'cache/feat_dev_20000_200.parquet').join(pl.read_parquet(O/'extra_dev.parquet'),on=['s1','src','eid'],validate='1:1',maintain_order='left')
nts['dev']=pl.read_parquet(B/'cache/ntrue_dev_20000_200.parquet')
dev_scores=pl.read_parquet(B/'cache/dev_scored.parquet').select('s1','src','eid','score')
early=frames['dev'].join(dev_scores,on=['s1','src','eid'],maintain_order='left')
hard=(early['label'].to_numpy()==1)|(early['score'].to_numpy()>=.05)
rng=np.random.default_rng(20260928);keep=hard|(rng.random(len(hard))<.02)
early_weight=np.where(hard[keep],1.,50.).astype('float32');early=early.filter(pl.Series(keep))
print('early-stopping rows',early.height,flush=True)
extra=[c for c in frames['train'].columns if c not in basefeatures+['s1','src','eid','label']]
configs={'sampled_baseline':basefeatures,'all_features':basefeatures+extra,'rarity_numeric_only':basefeatures+[f for f in extra if not f.startswith('mapped_') and f!='name_consonant_ratio']}
print('training after hard-negative retention and weighted easy-negative sampling',frames['train'].height,flush=True)
records=[]
for name,feats in configs.items():
 if (O/f'{name}_decision.json').exists():
  print('completed model already exists',name,flush=True);continue
 t=time.time();model=lgb.LGBMClassifier(objective='binary',learning_rate=.05,num_leaves=127,min_child_samples=100,colsample_bytree=.9,subsample=.8,subsample_freq=1,n_estimators=650,n_jobs=2,random_state=42,verbosity=-1)
 model.fit(frames['train'].select(feats).to_numpy(),frames['train']['label'].to_numpy(),sample_weight=weights,feature_name=feats,eval_set=[(early.select(feats).to_numpy(),early['label'].to_numpy())],eval_sample_weight=[early_weight],eval_metric='average_precision',callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(100)])
 model.booster_.save_model(str(O/f'{name}.txt'))
 s=frames['dev'].select('s1','src','eid','label').with_columns(pl.Series('score',model.predict_proba(frames['dev'].select(feats).to_numpy())[:,1].astype('float32')))
 results=E.sweep(s,nts['dev'],np.round(np.arange(.3,.951,.025),3));best=results.sort('macro_f05',descending=True).row(0,named=True);record={'experiment_id':'ASTRA003_'+name,'change':name,'validation_f05':best['macro_f05'],'precision':best['macro_precision'],'recall':best['macro_recall'],'singleton_accuracy':best['singleton_accuracy'],'candidate_recall':frames['dev']['label'].sum()/nts['dev']['ntrue'].sum(),'runtime':time.time()-t,'decision':'retain_for_further_validation' if best['macro_f05']>.9184494 else 'reject','reason':'development-only comparison on identical candidates','threshold':best['threshold']};records.append(record)
 model.booster_.save_model(str(O/f'{name}.txt'));s.write_parquet(O/f'{name}_dev_scored.parquet');(O/f'{name}_decision.json').write_text(json.dumps({'features':feats,'threshold':best['threshold'],'dev':best,'best_iteration':model.best_iteration_},indent=2));results.write_csv(O/f'{name}_thresholds.csv');pl.DataFrame(records).write_csv(O/'experiments.csv');print(record,flush=True)
