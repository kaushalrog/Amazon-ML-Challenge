"""Read-only analysis of the protected local snapshot; never promotes a submission."""
import sys,json,time,csv
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[2]; B=R/'experiments/astra_baseline'; C=B/'cache'; O=R/'reports'
sys.path.insert(0,str(B/'src'))
import evaluate_v2 as E
start=time.time(); decision=json.loads((B/'models/decision.json').read_text()); threshold=decision['threshold']; model=lgb.Booster(model_file=str(B/'models/matcher_lgbm.txt'))
metrics={}; sets={}
for fold,n in [('train',60000),('dev',20000),('eval',20000)]:
 nt=pl.read_parquet(C/f'ntrue_{fold}_{n}_200.parquet');sets[fold]=set(nt['s1'])
 if fold=='train':continue
 x=pl.read_parquet(C/f'feat_{fold}_{n}_200.parquet'); cached=pl.read_parquet(C/f'{fold}_scored.parquet')
 s=x.select('s1','src','eid','label').with_columns(pl.Series('score',model.predict(x.select(decision['features']).to_numpy(),num_threads=2).astype('float32')))
 delta=s.join(cached,on=['s1','src','eid','label'],suffix='_cached').select((pl.col('score')-pl.col('score_cached')).abs().max()).item()
 metrics[fold]=E.evaluate(s,nt,threshold)|{'candidate_recall':int(x['label'].sum())/int(nt['ntrue'].sum()),'candidate_count':x.height,'max_replay_score_difference':delta,'duplicate_pairs':x.height-x.unique(subset=['s1','src','eid']).height}
 print(fold,metrics[fold],flush=True)
 if fold=='dev':dev=x.join(s.select('s1','src','eid','score'),on=['s1','src','eid']); devnt=nt
assert not (sets['train']&sets['dev'] or sets['train']&sets['eval'] or sets['dev']&sets['eval'])
metrics['split_overlap']=0
truth=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(devnt.select('s1'),on='s1',how='semi')
assert dev.filter(pl.col('label')==1).select('s1','src','eid').sort('s1','src','eid').equals(truth.join(dev.select('s1','src','eid'),on=['s1','src','eid'],how='semi').sort('s1','src','eid'))
source1=pl.read_parquet(R/'work/cache/train_source1.parquet').select(pl.col('eid').alias('s1'),'name','addr',pl.col('country').cast(pl.String))
selected=source1.join(devnt.select('s1'),on='s1',how='semi')
# Audit duplicate normalized name+address signatures across actual model train and dev samples.
trainrecords=source1.filter(pl.col('s1').is_in(list(sets['train'])))
metrics['dev_exact_signature_overlap_with_train']=selected.join(trainrecords.select('name','addr','country').unique(),on=['name','addr','country'],how='semi').height
D={};T={i:[] for i in devnt['s1']}; country=dict(selected.select('s1','country').iter_rows())
for a,b,c in truth.iter_rows():T[a].append(f'S{b}-{c}')
for row in dev.sort(['s1','score'],descending=[False,True]).select('s1','src','eid','score','name_tsr','addr_tsr','num_conflict','name_missing','addr_missing').iter_rows():D.setdefault(row[0],[]).append(row[1:])
rows=[]; clusters={}; fp=fn=blocked=classified=0
for sid in devnt['s1']:
 rr=D.get(sid,[]); ids=[f'S{x[0]}-{x[1]}' for x in rr]; pred=[ids[i] for i,x in enumerate(rr) if x[2]>=threshold]; true=set(T[sid]); pp=set(pred); cc=set(ids)
 nf=len(pp-true); nn=len(true-pp); nb=len(true-cc); nc=len((true&cc)-pp)
 kind=('Mixed FP/FN' if nf and nn else 'False positive' if nf else 'False negative' if nn else 'Correct singleton' if not true else 'Correct one-match' if len(true)==1 else 'Correct multi-match')
 tags=[]
 if nb:tags.append('candidate_blocking_miss')
 if nc:tags.append('candidate_generated_rejected')
 if nf and not true:tags.append('singleton_false_match')
 for i,x in enumerate(rr):
  if (ids[i] in pp-true) or (ids[i] in true-pp):
   if x[3]>=.9 and x[4]<.5:tags.append('strong_name_weak_address')
   if x[4]>=.9 and x[3]<.5:tags.append('strong_address_weak_name')
   if x[5]:tags.append('numeric_conflict')
   if x[6]:tags.append('missing_name')
   if x[7]:tags.append('missing_address')
   tags.append(f'S{x[0]}_candidate_error')
 if nf or nn:tags.append('country_'+country[sid])
 for tag in set(tags):clusters[tag]=clusters.get(tag,0)+1
 fp+=nf;fn+=nn;blocked+=nb;classified+=nc
 top=rr[0] if rr else None; second=rr[1] if len(rr)>1 else None
 rows.append({'source1_entity_id':f'S1-{sid}','true_ids':','.join(sorted(true)),'predicted_ids':','.join(pred),'candidate_ids':','.join(ids),'model_scores':json.dumps([round(x[2],7) for x in rr]),'top_candidate':ids[0] if ids else '', 'second_candidate':ids[1] if len(ids)>1 else '', 'score_gap':top[2]-(second[2] if second else 0) if top else None,'name_similarities':json.dumps([round(x[3],5) for x in rr]),'address_similarities':json.dumps([round(x[4],5) for x in rr]),'country':country[sid],'source_pair':','.join(sorted({f'S1-S{x[0]}' for x in rr})),'error_type':kind,'error_clusters':','.join(sorted(set(tags))),'false_positives':nf,'false_negatives':nn,'blocking_misses':nb,'classified_false_negatives':nc})
pl.DataFrame(rows).write_csv(O/'error_cases.tsv',separator='\t')
pl.DataFrame([{'cluster':k,'affected_entities':v,'note':'overlapping diagnostic tags; not mutually exclusive'} for k,v in sorted(clusters.items(),key=lambda x:-x[1])]).write_csv(O/'error_cluster_summary.csv')
metrics['dev_error_counts']={'fp_pairs':fp,'fn_pairs':fn,'blocked_true_pairs':blocked,'generated_but_rejected_true_pairs':classified,'entity_outcomes':{k:sum(r['error_type']==k for r in rows) for k in set(r['error_type'] for r in rows)}}
# Every blocked-out true pair with both normalized records, for systematic inspection.
miss=truth.join(dev.select('s1','src','eid'),on=['s1','src','eid'],how='anti').join(selected,on='s1')
right=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{s}.parquet').join(miss.lazy().filter(pl.col('src')==s).select('src','eid').unique(),on=['src','eid'],how='semi').collect() for s in [2,3]])
miss=miss.join(right.rename({'name':'candidate_name','addr':'candidate_addr','country':'candidate_country'}),on=['src','eid'])
miss.with_columns(pl.col('candidate_country').cast(pl.String)).write_csv(O/'astra_blocked_true_pairs.tsv',separator='\t')
groups=[]
for src in [2,3]:
 ss=dev.filter(pl.col('src')==src); tt=truth.filter(pl.col('src')==src)
 nt=devnt.select('s1').join(tt.group_by('s1').len().rename({'len':'ntrue'}),on='s1',how='left').fill_null(0)
 groups.append({'group':f'S{src}',**E.evaluate(ss,nt,threshold),'candidate_recall':ss['label'].sum()/tt.height})
for country_value in selected['country'].unique():
 ids=selected.filter(pl.col('country')==country_value).select('s1');ss=dev.join(ids,on='s1',how='semi');nt=devnt.join(ids,on='s1',how='semi')
 groups.append({'group':country_value,**E.evaluate(ss,nt,threshold),'candidate_recall':ss['label'].sum()/nt['ntrue'].sum()})
pl.DataFrame(groups).write_csv(O/'astra_source_country_metrics.csv')
# Dev-only policy experiments. Eval replay above is baseline reproduction, never selection.
experiments=[]
def record(name,ss,t,reason):
 m=E.evaluate(ss,devnt,t);experiments.append({'experiment_id':name,'change':reason,'validation_f05':m['macro_f05'],'validation_precision':m['macro_precision'],'validation_recall':m['macro_recall'],'singleton_accuracy':m['singleton_accuracy'],'candidate_recall':metrics['dev']['candidate_recall'],'leaderboard_score':None,'runtime':time.time()-start,'kept':'NO','reason':'dev diagnostic only; negligible gain or regression; no promotion','threshold':t});return m
record('ASTRA_LOCAL_BASELINE',dev,threshold,'Protected local model replay; external benchmark 99.08 belongs to another team')
for t in np.round(np.arange(.30,.951,.05),2):record(f'GLOBAL_{t}',dev,float(t),'global threshold')
r=dev.with_columns(pl.col('score').rank('ordinal',descending=True).over('s1').alias('rank'),pl.col('score').max().over('s1').alias('top'))
second=r.filter(pl.col('rank')==2).select('s1',pl.col('score').alias('second'));r=r.join(second,on='s1',how='left').with_columns(pl.col('second').fill_null(0.))
for t in [.5,.62,.75,.9]:
 record(f'TOP1_{t}',r.filter(pl.col('rank')==1),t,'top1 diagnostic')
 for margin in [.01,.05,.1]:record(f'MARGIN_{t}_{margin}',r.filter((pl.col('rank')==1)&(pl.col('top')-pl.col('second')>=margin)),t,'top1 plus score margin')
 for frac in [.7,.9,.95]:record(f'RELATIVE_{t}_{frac}',r.filter(pl.col('score')>=pl.col('top')*frac),t,'multi-match relative floor')
for t2 in [.5,.62,.75]:
 for t3 in [.5,.62,.75]:record(f'SOURCE_{t2}_{t3}',dev.filter(((pl.col('src')==2)&(pl.col('score')>=t2))|((pl.col('src')==3)&(pl.col('score')>=t3))),0.,f'source-specific thresholds S2={t2}, S3={t3}')
for cut in [.3,.5,.7]:record(f'CONTRADICTION_{cut}',dev.filter(~((pl.col('num_conflict')==1)&(pl.col('addr_tsr')<cut))),threshold,'reject numeric-conflict/weak-address candidates')
pl.DataFrame(experiments).write_csv(O/'astra_experiments.csv')
metrics['best_dev_policy']=max(experiments,key=lambda x:x['validation_f05']);metrics['elapsed_seconds']=time.time()-start
(O/'astra_reproduction.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2),flush=True)
