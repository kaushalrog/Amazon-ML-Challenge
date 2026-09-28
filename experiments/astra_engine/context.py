"""Label-free relative evidence within each S1 and source candidate group."""
import polars as pl
GROUPS=['s1','src']
COLUMNS=['name_x_addr','joint_retrieval','mapped_name_tsr','name_idf_coverage','addr_tsr']
FEATURES=[f'{c}_behind_best' for c in COLUMNS]+['name_exact_count','addr_exact_count']
def add(x):
 return x.with_columns(*[(pl.col(c).max().over(GROUPS)-pl.col(c)).cast(pl.Float32).alias(f'{c}_behind_best') for c in COLUMNS],*[pl.col(c).sum().over(GROUPS).cast(pl.Float32).alias(f'{c}_count') for c in ['name_exact','addr_exact']])
