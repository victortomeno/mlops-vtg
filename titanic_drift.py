from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

DATA_DIR = Path("data/titanic")
REPORTS_DIR = Path("reports/titanic")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STRATIFY_OPTIONS = [True, False]
SPLIT_RATIOS = [
    (0.60, 0.20, 0.20),
    (0.90, 0.05, 0.05),
    (0.98, 0.01, 0.01),
]
SEEDS = [0, 13]

TITANIC_NUMERICAL = ["Age", "SibSp", "Parch", "Fare"]
TITANIC_CATEGORICAL = ["Pclass", "Sex", "Embarked", "Survived"]
ID_COLUMN = "PassengerId"

titanic_schema = DataDefinition(
    numerical_columns=TITANIC_NUMERICAL,
    categorical_columns=TITANIC_CATEGORICAL,
    id_column=ID_COLUMN,
)

def build_report() -> Report:
    return Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset(),
        ],
    )

def run_single_comparison(
    condition_name: str,
    train_df: pd.DataFrame,
    check_df: pd.DataFrame,
    check_label: str,
) -> dict:
    ref_dataset = Dataset.from_pandas(train_df, data_definition=titanic_schema)
    cur_dataset = Dataset.from_pandas(check_df, data_definition=titanic_schema)

    report = build_report()
    result = report.run(current_data=cur_dataset, reference_data=ref_dataset)

    html_path = REPORTS_DIR / f"{condition_name}_{check_label}.html"
    json_path = REPORTS_DIR / f"{condition_name}_{check_label}.json"

    result.save_html(str(html_path))
    result.save_json(str(json_path))

    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
        "condition": condition_name,
        "check": check_label,
    }

def main():
    results = []

    for stratify in STRATIFY_OPTIONS:
        for train_r, val_r, test_r in SPLIT_RATIOS:
            for seed in SEEDS:
                strat_label = "stratified" if stratify else "no_stratified"
                ratio_label = f"{int(train_r*100)}_{int(val_r*100)}_{int(test_r*100)}"
                condition_name = f"{ratio_label}_seed{seed}_{strat_label}"

                data_dir = DATA_DIR / condition_name
                train_df = pd.read_csv(data_dir / "train.csv")
                val_df = pd.read_csv(data_dir / "val.csv")
                test_df = pd.read_csv(data_dir / "test.csv")

                print(f"Processing: {condition_name}")

                res_val = run_single_comparison(condition_name, train_df, val_df, "val")
                results.append(res_val)

                res_test = run_single_comparison(condition_name, train_df, test_df, "test")
                results.append(res_test)

                print(f"  Saved: {res_val['html_path']}")
                print(f"  Saved: {res_test['html_path']}")

    print(f"\nDone! Generated {len(results)} drift reports.")
    return results

if __name__ == "__main__":
    main()