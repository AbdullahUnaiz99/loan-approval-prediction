import yaml, joblib
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from data_preprocessing import create_preprocessor

CONFIG = yaml.safe_load(Path("config/config.yaml").read_text())
TARGET = CONFIG["model"]["target"]

def train_model(df: pd.DataFrame):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    preprocessor = create_preprocessor(df, TARGET)

    pipe = ImbPipeline(
        steps=[
            ("pre", preprocessor),
            ("smote", SMOTE()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, random_state=42
                ),
            ),
        ]
    )

    param_grid = {
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=CONFIG["evaluation"]["cv_folds"])
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring=CONFIG["evaluation"]["primary_metric"],
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X, y)

    best_model_path = (
        Path(CONFIG["model"]["save_dir"]) / "best_model.joblib"
    )
    best_model_path.parent.mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_, best_model_path)
    print(f"Saved model to {best_model_path}")
