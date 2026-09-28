"""Additional spelling evidence and macro-aware weighting: dev-only comparisons."""
from pathlib import Path
import sys,json,time
import polars as pl,numpy as np,lightgbm as lgb
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;P=R/'experiments/astra_011';sys.path.insert(0,str(R/'experiments/astra_baseline/src'));import evaluate_v2 as E
sys.path.insert(0,str(R/'experiments/astra_engine'));import context as C,expanded_map as M
key=['s1','src','eid'];tr=pl.read_parquet(P/'mined_train.parquet').join(pl.read_parquet(O/'train_extra.parquet'),on=key,validate='1:1',maintain_order='left').join(pl.read_parquet(P/'train/ntrue.parquet'),on='s1',maintain_order='left');dev=C.add(pl.read_parquet(P/'dev/parts/*.parquet')).join(pl.read_parquet(O/'dev_extra.parquet'),on=key,validate='1:1');nt=pl.read_parquet(P/'dev/ntrue.parquet');cfg=json.loads((P/'broader_retrieval_decision.json').read_text());base=cfg['features'];old=lgb.Booster(model_file=str(P/'broader_retrieval.txt'));pred=old.predict(dev.select(base).to_numpy(),num_threads=2);hard=(dev['label'].to_numpy()==1)|(pred>=.02);rng=np.random.default_rng(20260930);keep=hard|(rng.random(dev.height)<.1);early=dev.filter(pl.Series(keep));ew=np.where(hard[keep],1.,10.);rows=pl.read_csv(O/'experiments.csv').to_dicts() if (O/'experiments.csv').exists() else []
for name,features,power,trees in [('expanded_map',base+M.FEATURES,0.,1000),('expanded_context',base+C.FEATURES+M.FEATURES,0.,1400),('expanded_macro',base+C.FEATURES+M.FEATURES,.5,1400)]:
 if (O/f'{name}_decision.json').exists():
  print('completed',name,flush=True);continue
 t=time.time();weights=tr['_weight'].to_numpy()/np.maximum(tr['ntrue'].to_numpy(),1)**power;m=lgb.LGBMClassifier(objective='binary',learning_rate=.045,num_leaves=127,min_child_samples=80,colsample_bytree=.9,subsample=.85,subsample_freq=1,n_estimators=trees,n_jobs=2,random_state=20260929,verbosity=-1)
 if (O/f'{name}.txt').exists():
  booster=lgb.Booster(model_file=str(O/f'{name}.txt'))
 else:
  m.fit(tr.select(features).to_numpy(),tr['label'].to_numpy(),sample_weight=weights,feature_name=features,eval_set=[(early.select(features).to_numpy(),early['label'].to_numpy())],eval_sample_weight=[ew],eval_metric='average_precision',callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(200)])
  booster=m.booster_;booster.save_model(str(O/f'{name}.txt'))
 s=dev.select(*key,'label').with_columns(pl.Series('score',booster.predict(dev.select(features).to_numpy(),num_threads=2).astype('float32')));s.write_parquet(O/f'{name}_dev_scored.parquet');sw=E.sweep(s,nt,np.round(np.arange(.3,.951,.025),3));sw.write_csv(O/f'{name}_thresholds.csv');best=sw.sort('macro_f05',descending=True).row(0,named=True);out=dict(cfg);out.update(features=features,threshold=best['threshold'],dev=best,weight_power=power,expanded_map=True,best_iteration=booster.current_iteration());(O/f'{name}_decision.json').write_text(json.dumps(out,indent=2));rows.append(dict(experiment_id=name,change=name,validation_f05=best['macro_f05'],precision=best['macro_precision'],recall=best['macro_recall'],singleton_accuracy=best['singleton_accuracy'],candidate_recall=dev['label'].sum()/nt['ntrue'].sum(),runtime=time.time()-t,decision='development comparison',reason='train-only lexicon; all hard negatives retained',threshold=best['threshold']));pl.DataFrame(rows).write_csv(O/'experiments.csv');print(rows[-1],flush=True)
