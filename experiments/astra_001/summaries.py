from pathlib import Path
import json,time
import polars as pl
R=Path(__file__).resolve().parents[2];O=R/'reports';B=R/'experiments/astra_baseline/cache'
x=pl.read_parquet(B/'feat_dev_20000_200.parquet').join(pl.read_parquet(B/'dev_scored.parquet').select('s1','src','eid','score'),on=['s1','src','eid'])
rows=[]
for group,df in [('positive_generated',x.filter(pl.col('label')==1)),('hard_negative_score_above_0.1',x.filter((pl.col('label')==0)&(pl.col('score')>.1))),('false_positive',x.filter((pl.col('label')==0)&(pl.col('score')>=.62))),('rejected_positive',x.filter((pl.col('label')==1)&(pl.col('score')<.62)))]:
 for f in ['name_tsr','name_jac','addr_tsr','addr_jac','num_jac','num_conflict','name_x_addr','addr_missing','score']:
  a=df[f]; rows.append({'group':group,'feature':f,'count':df.height,'mean':a.mean(),'p10':a.quantile(.1),'median':a.median(),'p90':a.quantile(.9)})
pl.DataFrame(rows).write_csv(O/'astra_signal_distributions.csv')
bins=x.with_columns((pl.col('score')*10).floor().clip(0,9).alias('bin')).group_by('bin').agg(pl.len().alias('pairs'),pl.col('score').mean().alias('mean_score'),pl.col('label').mean().alias('positive_fraction')).sort('bin');bins.write_csv(O/'astra_calibration.csv')
e=pl.read_csv(O/'error_cases.tsv',separator='\t',infer_schema_length=10000).with_columns(pl.col('true_ids').fill_null(''),pl.col('predicted_ids').fill_null(''))
e=e.with_columns(pl.when(pl.col('true_ids')=='').then(0).otherwise(pl.col('true_ids').str.count_matches(',')+1).alias('ntrue'),pl.when(pl.col('predicted_ids')=='').then(0).otherwise(pl.col('predicted_ids').str.count_matches(',')+1).alias('npred'))
e=e.with_columns((pl.col('ntrue')-pl.col('false_negatives')).alias('tp')).with_columns(pl.when(pl.col('ntrue')==0).then((pl.col('npred')==0).cast(pl.Float64)).otherwise(1.25*pl.col('tp')/(.25*pl.col('ntrue')+pl.col('npred'))).alias('f05'))
e.with_columns(pl.when(pl.col('ntrue')>=3).then(pl.lit('3+')).otherwise(pl.col('ntrue').cast(pl.String)).alias('truth_cardinality')).group_by('truth_cardinality').agg(pl.len().alias('entities'),pl.col('f05').mean(),pl.col('false_positives').sum(),pl.col('false_negatives').sum(),pl.col('blocking_misses').sum()).sort('truth_cardinality').write_csv(O/'astra_multimatch.csv')
# Metric loss contributions: oracle fixes, diagnostic only, never prediction policies.
base=e['f05'].mean();out=[]
for name,npred,tp in [('recover_blocked_true_matches',pl.col('npred')+pl.col('blocking_misses'),pl.col('tp')+pl.col('blocking_misses')),('recover_rejected_true_matches',pl.col('npred')+pl.col('classified_false_negatives'),pl.col('tp')+pl.col('classified_false_negatives')),('remove_false_merges',pl.col('npred')-pl.col('false_positives'),pl.col('tp'))]:
 f=pl.when(pl.col('ntrue')==0).then((npred==0).cast(pl.Float64)).otherwise(1.25*tp/(.25*pl.col('ntrue')+npred));score=e.select(f.mean()).item();out.append({'oracle_intervention':name,'macro_f05':score,'gain':score-base,'share_of_baseline_score_loss':(score-base)/(1-base)})
pl.DataFrame(out).write_csv(O/'astra_error_loss.csv')
print('Error summaries complete',flush=True)
# Raw supplied TSV data, sequential streaming to limit peak memory.
stats=[]
for split in ['train','test']:
 for source in [1,2,3]:
  path=R/f'dataset/{split}/{split}_source{source}.tsv';q=pl.scan_csv(path,separator='\t',quote_char=None,infer_schema_length=0)
  a=q.select(pl.len().alias('rows'),pl.col('entity_id').n_unique().alias('unique_ids'),*[pl.col(c).fill_null('').eq('').sum().alias(c+'_missing') for c in ['business_name','business_address','country']],*[pl.col(c).fill_null('').str.len_chars().mean().alias(c+'_mean_length') for c in ['business_name','business_address']],pl.col('business_name').n_unique().alias('unique_names'),pl.col('business_address').n_unique().alias('unique_addresses'),pl.struct('business_name','business_address','country').n_unique().alias('unique_name_address_country'),pl.col('business_address').fill_null('').str.contains(r'\d').mean().alias('address_numeric_rate'),pl.col('entity_id').str.contains(r'^S[123]-0\d').sum().alias('leading_zero_ids')).collect(engine='streaming').row(0,named=True)
  a.update(split=split,source=source,countries=q.group_by('country').len().collect().to_dicts());stats.append(a);print(split,source,a['rows'],flush=True)
  (O/'astra_dataset_stats.json').write_text(json.dumps(stats,indent=2))
gt=pl.scan_csv(R/'dataset/train/train_ground_truth.tsv',separator='\t',quote_char=None,infer_schema_length=0).with_columns(pl.col('matched_entity_ids').fill_null('').alias('ids')).with_columns(pl.when(pl.col('ids')=='').then(0).otherwise(pl.col('ids').str.count_matches(',')+1).alias('n'))
g=gt.select(pl.len().alias('s1_entities'),(pl.col('n')==0).sum().alias('singletons'),(pl.col('n')==1).sum().alias('one_match'),(pl.col('n')>1).sum().alias('multi_match'),pl.col('n').mean().alias('mean_matches'),pl.col('n').max().alias('max_matches'),pl.col('ids').str.contains('S2-').mean().alias('s2_match_rate'),pl.col('ids').str.contains('S3-').mean().alias('s3_match_rate'),(pl.col('ids').str.contains('S2-')&pl.col('ids').str.contains('S3-')).mean().alias('both_source_rate')).collect().row(0,named=True);(O/'astra_ground_truth_stats.json').write_text(json.dumps(g,indent=2));print(g,flush=True)
