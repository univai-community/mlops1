from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn import base
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from hamilton.function_modifiers import config, extract_fields


@extract_fields({'X_train': pd.DataFrame, 'X_test': pd.DataFrame, 'y_train': pd.Series, 'y_test': pd.Series})
def train_test_split_func(
    final_imputed_features: pd.DataFrame,  # feature matrix
    target: pd.Series,  # the target or the y
    validation_size_fraction: float,  # the validation fraction
    random_state: int,  # random state for reproducibility
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:  # dictionary of dataframes and Series
    """Function that creates the training & test splits.
    It this then extracted out into constituent components and used downstream.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        final_imputed_features, target, test_size=validation_size_fraction, stratify=target, random_state=random_state
    )
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


# no @config.when needed because we have only one classifier
def prefit_clf(
    random_state: int,  # get random state from parameters
    # either None or given max_depth hyperparameter
    max_depth: Union[int, None]
) -> base.ClassifierMixin:  # return an unfit Random Forest
    return RandomForestClassifier(max_depth=max_depth, random_state=random_state)


def fit_clf(
    prefit_clf: base.ClassifierMixin,  # prefit classifier
    X_train: pd.DataFrame,  # transformed features matrix
    y_train: pd.Series,  # target column
) -> base.ClassifierMixin:
    """Calls fit on the classifier object; it mutates the classifier and fits it."""
    prefit_clf.fit(X_train, y_train)
    return prefit_clf


def train_predictions(
    fit_clf: base.ClassifierMixin,  # already fit classifier
    X_train: pd.DataFrame,  # training or testing dataframe
    t: float = 0.5  # classification probability threshold
) -> Tuple[float, int]:  # Probabilities from model, Predictions from model
    y_proba = fit_clf.predict_proba(X_train)[:, 1]
    y_preds = 1*(y_proba >= t)
    return y_proba, y_preds


def test_predictions(
    fit_clf: base.ClassifierMixin,  # already fit classifier
    X_test: pd.DataFrame,  # training or testing dataframe
    t: float = 0.5  # classification probability threshold
) -> Tuple[float, int]:  # Probabilities from model, Predictions from model
    y_proba = fit_clf.predict_proba(X_test)[:, 1]
    y_preds = 1*(y_proba >= t)
    return y_proba, y_preds
