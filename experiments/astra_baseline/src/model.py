import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from features import extract_features
from validation import evaluate_macro_f05
from blocking import load_split_data
from normalize import normalize_df
import json

def generate_training_data(s1_df, cand_df, y_true):
    from blocking import generate_candidate_pairs
    cand_pairs = generate_candidate_pairs(s1_df, cand_df)
    
    # Combine and create pairs
    records = []
    
    cand_df_idx = cand_df.set_index('entity_id')
    
    for _, s1_row in s1_df.iterrows():
        s1 = s1_row['entity_id']
        cands = cand_pairs.get(s1, set())
        
        true_matches = y_true.get(s1, set())
        
        import random
        random.seed(42)
        
        cands_list = list(cands)
        positives = [c for c in cands_list if c in true_matches]
        negatives = [c for c in cands_list if c not in true_matches]
        
        sampled_negatives = random.sample(negatives, min(15, len(negatives)))
        
        for c in positives + sampled_negatives:
            is_match = 1 if c in true_matches else 0
            
            # extract features safely (handle Series if multiple matches exist with same id, though entity_ids should be unique)
            c_row_df = cand_df_idx.loc[[c]] if c in cand_df_idx.index else None
            if c_row_df is not None and len(c_row_df) > 0:
                c_row = c_row_df.iloc[0]
                features = extract_features(s1_row, c_row)
                
                rec = {
                    's1': s1,
                    'c': c,
                    'label': is_match
                }
                rec.update(features)
                records.append(rec)
            
    return pd.DataFrame(records)

def train_lightgbm(train_df):
    features = [c for c in train_df.columns if c not in ['s1', 'c', 'label']]
    print("Features:", features)
    
    # Group by s1
    gkf = GroupKFold(n_splits=5)
    
    oof_preds = np.zeros(len(train_df))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df['s1'])):
        print(f"Fold {fold}")
        
        X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx]['label']
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['label']
        
        # Weighted to handle imbalance
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0
        
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            class_weight={0: 1.0, 1: pos_weight},
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)
        
    train_df['pred_prob'] = oof_preds
    
    # Feature importance
    importance = np.mean([m.feature_importances_ for m in models], axis=0)
    fi_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)
    fi_df.to_csv('reports/feature_importance.csv', index=False)
    
    return train_df, models, fi_df

def optimize_threshold(scored_df, y_true):
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    results = []
    
    for th in thresholds:
        # predicted dict
        pred_dict = {}
        for s1, group in scored_df.groupby('s1'):
            cands = group[group['pred_prob'] >= th]['c'].tolist()
            pred_dict[s1] = set(cands)
            
        res = evaluate_macro_f05(y_true, pred_dict)
        res['threshold'] = th
        results.append(res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("reports/threshold_search.csv", index=False)
    
    best_th = df_res.loc[df_res['macro_f05'].idxmax()]['threshold']
    print(f"Best threshold: {best_th}")
    print(df_res.loc[df_res['macro_f05'].idxmax()])
    
    return best_th, df_res

def main():
    print("Loading data...")
    s1_val, cand_df, y_true = load_split_data()
    
    print("Normalizing...")
    s1_val = normalize_df(s1_val)
    cand_df = normalize_df(cand_df)
    
    print("Generating training data (Blocking + Features)...")
    train_df = generate_training_data(s1_val, cand_df, y_true)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Positive ratio: {train_df['label'].mean():.4f}")
    
    print("Training LightGBM...")
    scored_df, models, fi_df = train_lightgbm(train_df)
    
    print("Optimizing threshold...")
    best_th, df_res = optimize_threshold(scored_df, y_true)
    
    # Save the first model
    models[0].booster_.save_model('models/lightgbm_model.txt')
    
    with open("reports/decision_analysis.md", "w") as f:
        f.write("# Decision Threshold Analysis\n\n")
        f.write(f"Best threshold: {best_th}\n\n")
        f.write(df_res.to_html(index=False))

if __name__ == "__main__":
    main()
