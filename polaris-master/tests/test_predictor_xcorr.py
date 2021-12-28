"""
`pytest` testing framework file for xcorr predictor
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from polaris.learn.predictor.cross_correlation import XCorr
from polaris.learn.predictor.cross_correlation_configurator import \
    CrossCorrelationConfigurator


def test_xcorr():
    """
    `pytest` entry point
    """

    test_df = pd.DataFrame({
        "A": [4, 123, 24.2, 3.14, 1.41],
        "B": [7, 0, 24.2, 3.14, 8.2]
    })

    configurator = CrossCorrelationConfigurator()
    parameters = configurator.get_configuration()
    metadata = {"analysis": {"column_tags": {}, "feature_columns": None}}
    correlator = XCorr(metadata, parameters)
    assert correlator.importances_map is None

    correlator.fit(test_df)
    assert correlator.importances_map is not None
    assert isinstance(correlator.importances_map, pd.DataFrame)
    assert correlator.importances_map.shape[0] == 2
    assert (correlator.importances_map.shape[1] ==
            correlator.importances_map.shape[0])


def test_xcorr_no_feature_columns():
    """
    `pytest` entry point
    """

    test_df = pd.DataFrame({
        "A": [4, 123, 24.2, 3.14, 1.41],
        "B": [7, 0, 24.2, 3.14, 8.2]
    })

    configurator = CrossCorrelationConfigurator()
    parameters = configurator.get_configuration()
    metadata = {}
    correlator = XCorr(metadata, parameters)
    assert correlator.importances_map is None

    correlator.fit(test_df)
    assert correlator.importances_map is not None
    assert isinstance(correlator.importances_map, pd.DataFrame)
    assert correlator.importances_map.shape[0] == 2
    assert (correlator.importances_map.shape[1] ==
            correlator.importances_map.shape[0])


def test_xcorr_pipeline():
    """
    `pytest` entry point
    """
    configurator = CrossCorrelationConfigurator()
    parameters = configurator.get_configuration()
    metadata = {"analysis": {"column_tags": {}, "feature_columns": None}}
    pipeline = Pipeline([("deps", XCorr(metadata, parameters))])

    assert pipeline is not None


def test_gridsearch_happy():
    """
    Test happy path for gridsearch
    """
    test_df = pd.DataFrame({
        "A": [4, 123, 24.2, 3.14, 1.41],
        "B": [7, 0, 24.2, 3.14, 8.2]
    })

    configurator = CrossCorrelationConfigurator(use_gridsearch=True)
    parameters = configurator.get_configuration()
    metadata = {"analysis": {"column_tags": {}, "feature_columns": None}}

    parameters.test_size = 0.1
    parameters.gridsearch_n_splits = 2

    correlator = XCorr(metadata, parameters)
    correlator.fit(test_df)
    assert correlator.importances_map is not None
    assert isinstance(correlator.importances_map, pd.DataFrame)
    assert correlator.importances_map.shape[0] == 2
    assert (correlator.importances_map.shape[1] ==
            correlator.importances_map.shape[0])


def test_gridsearch_incompatible_input():
    """
    Test incompatible input for gridsearch
    """
    test_df = [1, 2, 3, 4]

    configurator = CrossCorrelationConfigurator(use_gridsearch=True)
    parameters = configurator.get_configuration()
    metadata = {"analysis": {"column_tags": {}, "feature_columns": None}}

    correlator = XCorr(metadata, parameters)
    with pytest.raises(TypeError):
        correlator.fit(test_df)
