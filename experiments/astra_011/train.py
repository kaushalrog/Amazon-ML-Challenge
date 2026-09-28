"""Compare models on the complete new candidate distribution, selecting on dev only."""
from pathlib import Path
import sys,time,json
import numpy as np
import polars as pl
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R/'experiments/astra_baseline/src'));import evaluate_v2 as E
sys.path.insert(0,str(R/'experiments/astra_engine'));import context as C
oldcfg=json.loads((R/'experiments/astra_003/all_features_decision.json').read_text());old=lgb.Booster(model_file=str(R/'experiments/astra_003/all_features.txt'));base=oldcfg['features'];nt=pl.read_parquet(O/'dev/ntrue.parquet');dev=C.add(pl.read_parquet(O/'dev/parts/*.parquet'));rng=np.random.default_rng(20260929);t=time.time();records=[]
def measure(name,score,feats,runtime):
 s=dev.select('s1','src','eid','label').with_columns(pl.Series('score',score.astype('float32')));sw=E.sweep(s,nt,np.round(np.arange(.3,.951,.025),3));best=sw.sort('macro_f05',descending=True).row(0,named=True);s.write_parquet(O/f'{name}_dev_scored.parquet');sw.write_csv(O/f'{name}_thresholds.csv');cfg={'features':feats,'threshold':best['threshold'],'dev':best,'candidate_cap':400,'fast_filter':.0001,'retrieval_k':100};(O/f'{name}_decision.json').write_text(json.dumps(cfg,indent=2));record={'experiment_id':'ASTRA011_'+name,'change':name,'validation_f05':best['macro_f05'],'precision':best['macro_precision'],'recall':best['macro_recall'],'singleton_accuracy':best['singleton_accuracy'],'candidate_recall':dev['label'].sum()/nt['ntrue'].sum(),'runtime':runtime,'decision':'development comparison','reason':'expanded real candidates, full S1 denominators','threshold':best['threshold']};records.append(record);pl.DataFrame(records).write_csv(O/'experiments.csv');print(record,flush=True)
oldp=old.predict(dev.select(base).to_numpy(),num_threads=2);measure('previous_model',oldp,base,time.time()-t)
train=[]
for f in sorted((O/'train/parts').glob('*.parquet')):
 z=C.add(pl.read_parquet(f));pr=old.predict(z.select(base).to_numpy(),num_threads=2);hard=(z['label'].to_numpy()==1)|(pr>=.02);keep=hard|(rng.random(z.height)<.2);train.append(z.filter(pl.Series(keep)).with_columns(pl.Series('_weight',np.where(hard[keep],1.,5.).astype('float32'))))
tr=pl.concat(train);del train;tr.write_parquet(O/'mined_train.parquet');print('train rows',tr.height,'positives',tr['label'].sum(),flush=True)
hard=(dev['label'].to_numpy()==1)|(oldp>=.02);keep=hard|(rng.random(dev.height)<.1);early=dev.filter(pl.Series(keep));ew=np.where(hard[keep],1.,10.)
for name,features in [('broader_43',base),('broader_retrieval',base+['joint_retrieval']),('relative_evidence',base+['joint_retrieval']+C.FEATURES)]:
 t=time.time();m=lgb.LGBMClassifier(objective='binary',learning_rate=.045,num_leaves=127,min_child_samples=80,colsample_bytree=.9,subsample=.85,subsample_freq=1,n_estimators=1000,n_jobs=2,random_state=20260929,verbosity=-1)
 m.fit(tr.select(features).to_numpy(),tr['label'].to_numpy(),sample_weight=tr['_weight'].to_numpy(),feature_name=features,eval_set=[(early.select(features).to_numpy(),early['label'].to_numpy())],eval_sample_weight=[ew],eval_metric='average_precision',callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(200)])
 m.booster_.save_model(str(O/f'{name}.txt'));score=m.predict_proba(dev.select(features).to_numpy())[:,1];measure(name,score,features,time.time()-t)
