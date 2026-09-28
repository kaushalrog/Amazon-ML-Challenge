import pandas as pd
import numpy as np

def calculate_f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate_macro_f05(y_true_dict, y_pred_dict):
    """
    y_true_dict: dict mapping s1_id -> set of matched s2/s3 ids
    y_pred_dict: dict mapping s1_id -> set of predicted s2/s3 ids
    """
    precisions = []
    recalls = []
    f05s = []
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    singleton_correct = 0
    singleton_total = 0
    
    false_merges = 0
    missed_matches = 0
    
    all_s1 = set(y_true_dict.keys()) | set(y_pred_dict.keys())
    
    for s1 in all_s1:
        true_matches = y_true_dict.get(s1, set())
        pred_matches = y_pred_dict.get(s1, set())
        
        tp = len(true_matches & pred_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        false_merges += fp
        missed_matches += fn
        
        if len(true_matches) == 0:
            singleton_total += 1
            if len(pred_matches) == 0:
                singleton_correct += 1
                precisions.append(1.0)
                recalls.append(1.0)
                f05s.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(0.0)
                f05s.append(0.0)
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = calculate_f05(p, r)
            precisions.append(p)
            recalls.append(r)
            f05s.append(f)
            
    macro_p = np.mean(precisions) if precisions else 0.0
    macro_r = np.mean(recalls) if recalls else 0.0
    macro_f05 = np.mean(f05s) if f05s else 0.0
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "false_merges": false_merges,
        "missed_matches": missed_matches,
        "total_predicted_matches": sum(len(v) for v in y_pred_dict.values()),
        "total_true_matches": sum(len(v) for v in y_true_dict.values())
    }

def create_grouped_split(gt_df, val_size=0.2, random_state=42):
    """
    Splits the S1 entities into train and validation sets.
    Returns: train_s1, val_s1 (lists of entity IDs)
    """
    s1_entities = gt_df["source1_entity_id"].unique()
    np.random.seed(random_state)
    np.random.shuffle(s1_entities)
    
    split_idx = int(len(s1_entities) * (1 - val_size))
    train_s1 = s1_entities[:split_idx]
    val_s1 = s1_entities[split_idx:]
    
    return train_s1, val_s1

if __name__ == "__main__":
    # Test
    y_true = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1", "S2-2"}}
    y_pred = {"S1-1": {"S2-1"}, "S1-2": {"S2-3"}, "S1-3": {"S3-1"}}
    print(evaluate_macro_f05(y_true, y_pred))
