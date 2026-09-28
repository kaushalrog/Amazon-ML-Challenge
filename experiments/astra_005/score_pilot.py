"""Score the actual broadened pilot candidate set; keep full truth denominators."""
from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
import polars as pl
import lightgbm as lgb
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;B=R/'experiments/astra_baseline';sys.path.insert(0,str(B/'src'))
import features_v2 as F
import evaluate_v2 as E
n=2000;ids=pl.read_parquet(B/'cache/ntrue_dev_20000_200.parquet').sort('s1').head(n);base=pl.read_parquet(B/'cache/dev_scored.parquet').join(ids.select('s1'),on='s1',how='semi');parts=[base.select('s1','src','eid')]+[pl.read_parquet(p) for p in O.glob('*.parquet') if p.stem in ['mapped_name','mapped_sorted','consonant_name','name_prefix_number','consonant_prefix_number','address_without_numbers','address_sorted_without_numbers']];cand=pl.concat(parts).unique();truth=pl.read_parquet(R/'work/cache/gt_pairs.parquet').join(ids.select('s1'),on='s1',how='semi')
s1=pl.read_parquet(R/'work/cache/train_source1.parquet').join(ids.select(pl.col('s1').alias('eid')),on='eid',how='semi');pool=pl.concat([pl.scan_parquet(R/f'work/cache/train_source{s}.parquet').join(cand.lazy().filter(pl.col('src')==s).select('eid').unique(),on='eid',how='semi').collect(engine='streaming') for s in [2,3]])
X=F.build(cand,s1,pool).join(truth.with_columns(pl.lit(1,dtype=pl.Int8).alias('label')),on=['s1','src','eid'],how='left').with_columns(pl.col('label').fill_null(0));X.write_parquet(O/'pilot_features.parquet');model=lgb.Booster(model_file=str(B/'models/matcher_lgbm.txt'));score=X.select('s1','src','eid','label').with_columns(pl.Series('score',model.predict(X.select(F.FEATURES).to_numpy(),num_threads=2).astype('float32')));score.write_parquet(O/'pilot_scored.parquet');a=E.evaluate(base,ids,.62);b=E.evaluate(score,ids,.62);out={'baseline_same_pilot':a,'expanded_blocker_frozen_model_threshold':b,'candidate_recall':X['label'].sum()/truth.height,'candidate_count':X.height,'f05_gain':b['macro_f05']-a['macro_f05'],'status':'development pilot only; not promoted'};(O/'pilot_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
spec=importlib.util.spec_from_file_location('astra_features',R/'experiments/astra_003/features.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M);M.compute(F.attach_records(cand,s1,pool),M.resources()).write_parquet(O/'pilot_extra.parquet')
