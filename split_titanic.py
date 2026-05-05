from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TITANIC_CSV = Path("../titanic-dataset.csv")
OUTPUT_DIR = Path("data/titanic")

STRATIFY_OPTIONS = [True, False]
SPLIT_RATIOS = [
    (0.60, 0.20, 0.20),
    (0.90, 0.05, 0.05),
    (0.98, 0.01, 0.01),
]
SEEDS = [0, 13]

COLUMNS_TO_DROP = ["Name", "Ticket", "Cabin"]

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify: bool,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    stratify_target = df["Survived"] if stratify else None
    rest_ratio = val_ratio + test_ratio
    val_of_rest = val_ratio / rest_ratio

    train_df, rest_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=stratify_target,
    )

    rest_stratify = rest_df["Survived"] if stratify else None

    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_of_rest,
        random_state=random_seed,
        stratify=rest_stratify,
    )

    return train_df, val_df, test_df

def main():
    df = pd.read_csv(TITANIC_CSV)
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    print(f"Loaded Titanic dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Survived distribution: {df['Survived'].value_counts().to_dict()}")

    for stratify in STRATIFY_OPTIONS:
        for train_r, val_r, test_r in SPLIT_RATIOS:
            for seed in SEEDS:
                strat_label = "stratified" if stratify else "no_stratified"
                ratio_label = f"{int(train_r*100)}_{int(val_r*100)}_{int(test_r*100)}"

                out_dir = OUTPUT_DIR / f"{ratio_label}_seed{seed}_{strat_label}"
                out_dir.mkdir(parents=True, exist_ok=True)

                train_df, val_df, test_df = split_dataset(
                    df, train_r, val_r, test_r, stratify, seed
                )

                train_df.to_csv(out_dir / "train.csv", index=False)
                val_df.to_csv(out_dir / "val.csv", index=False)
                test_df.to_csv(out_dir / "test.csv", index=False)

                print(
                    f"[{ratio_label}_seed{seed}_{strat_label}] "
                    f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
                )

    print("\nDone! All splits generated.")

if __name__ == "__main__":
    main()