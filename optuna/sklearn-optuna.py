# %%
from sklearn.impute import SimpleImputer
from optuna import Trial


def instantiate_numerical_simple_imputer(
    trial: Trial, fill_value: int = -1
) -> SimpleImputer:
    strategy = trial.suggest_categorical(
        "numerical_strategy", ["mean", "median", "most_frequent", "constant"]
    )
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


def instantiate_categorical_simple_imputer(
    trial: Trial, fill_value: str = "missing"
) -> SimpleImputer:
    strategy = trial.suggest_categorical(
        "categorical_strategy", ["most_frequent", "constant"]
    )
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


from category_encoders import WOEEncoder


def instantiate_woe_encoder(trial: Trial) -> WOEEncoder:
    params = {
        "sigma": trial.suggest_float("sigma", 0.001, 5),
        "regularization": trial.suggest_float("regularization", 0, 5),
        "randomized": trial.suggest_categorical("randomized", [True, False]),
    }
    return WOEEncoder(**params)


from sklearn.preprocessing import RobustScaler


def instantiate_robust_scaler(trial: Trial) -> RobustScaler:
    params = {
        "with_centering": trial.suggest_categorical("with_centering", [True, False]),
        "with_scaling": trial.suggest_categorical("with_scaling", [True, False]),
    }
    return RobustScaler(**params)


from sklearn.ensemble import ExtraTreesClassifier


def instantiate_extra_trees(trial: Trial) -> ExtraTreesClassifier:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "max_features": trial.suggest_float("max_features", 0, 1),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": -1,
        "random_state": 42,
    }
    return ExtraTreesClassifier(**params)


from sklearn.pipeline import Pipeline


def instantiate_numerical_pipeline(trial: Trial) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", instantiate_numerical_simple_imputer(trial)),
            ("scaler", instantiate_robust_scaler(trial)),
        ]
    )
    return pipeline


def instantiate_categorical_pipeline(trial: Trial) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", instantiate_categorical_simple_imputer(trial)),
            ("encoder", instantiate_woe_encoder(trial)),
        ]
    )
    return pipeline


from sklearn.compose import ColumnTransformer


def instantiate_processor(
    trial: Trial, numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:

    numerical_pipeline = instantiate_numerical_pipeline(trial)
    categorical_pipeline = instantiate_categorical_pipeline(trial)

    processor = ColumnTransformer(
        [
            ("numerical_pipeline", numerical_pipeline, numerical_columns),
            ("categorical_pipeline", categorical_pipeline, categorical_columns),
        ]
    )

    return processor


def instantiate_model(
    trial: Trial, numerical_columns: list[str], categorical_columns: list[str]
) -> Pipeline:

    processor = instantiate_processor(trial, numerical_columns, categorical_columns)
    extra_trees = instantiate_extra_trees(trial)

    model = Pipeline([("processor", processor), ("extra_trees", extra_trees)])

    return model


from typing import Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, make_scorer
from pandas import DataFrame, Series
import numpy as np


def objective(
    trial: Trial,
    X: DataFrame,
    y: np.ndarray | Series,
    numerical_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    random_state: int = 42,
) -> float:
    if numerical_columns is None:
        numerical_columns = [*X.select_dtypes(exclude=["object", "category"]).columns]

    if categorical_columns is None:
        categorical_columns = [*X.select_dtypes(include=["object", "category"]).columns]

    model = instantiate_model(trial, numerical_columns, categorical_columns)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    roc_auc_scorer = make_scorer(
        roc_auc_score, multi_class="ovo", response_method="predict_proba"
    )
    scores = cross_val_score(model, X, y, scoring=roc_auc_scorer, cv=kf)

    return np.min([np.mean(scores), np.median([scores])])


# %% testing
from sklearn.datasets import load_iris

data = load_iris()
import pandas as pd

X_train = pd.DataFrame(data.data, columns=data.feature_names)
y_train = data.target

from optuna import create_study

study = create_study(study_name="optimization", direction="maximize")

study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)

# %%
