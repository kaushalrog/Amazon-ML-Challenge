# Workspace Inventory

## Repository Structure
- `dataset/`: Contains all TSV files for training and testing.
  - `train/`
    - `train_source1.tsv` (200MB)
    - `train_source2.tsv` (489MB)
    - `train_source3.tsv` (503MB)
    - `train_ground_truth.tsv` (127MB)
  - `test/`
    - `test_source1.tsv` (175MB)
    - `test_source2.tsv` (509MB)
    - `test_source3.tsv` (506MB)
- `utils/`: Contains `validate_submission.py`
- `Documentation_template.md`: Template for final methodology document.
- `README.md`: Existing README.

## New Directories Created
- `src/`: For source code.
- `reports/`: For markdown reports and CSV stats.
- `models/`: For saving trained models.
- `output/`: For generated predictions (`matching_results.tsv`, `candidate_pairs.tsv`).
- `submissions/`: For tracking submissions.
- `code/business_entity_resolution/`: For the final reproducible pipeline structure.
