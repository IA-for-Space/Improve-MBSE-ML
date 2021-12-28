"""
`pytest` testing framework file for xcorr configurator
"""

import json
from unittest import mock

from polaris.learn.predictor.cross_correlation_configurator import \
    CrossCorrelationConfigurator


def test_get_default_configuration():
    """
    Test for getting default configuration
    """
    configurator = CrossCorrelationConfigurator()
    parameters = configurator.get_configuration()

    assert not parameters.use_gridsearch
    assert parameters.random_state == 42
    assert parameters.test_size == 0.2
    assert parameters.gridsearch_scoring == "neg_mean_squared_error"
    assert parameters.gridsearch_n_splits == 18
    assert parameters.model_cpu_params == configurator.MODEL_CPU_PARAMS
    assert parameters.model_params['objective'] == "reg:squarederror"
    assert parameters.model_params['n_estimators'] == 80
    assert parameters.model_params['learning_rate'] == 0.1
    assert parameters.model_params['n_jobs'] == -1
    assert parameters.model_params['max_depth'] == 8


def test_get_default_configuration_using_gridsearch():
    """
    Test for getting default configuration
    """
    configurator = CrossCorrelationConfigurator(use_gridsearch=True)
    parameters = configurator.get_configuration()

    assert parameters.use_gridsearch
    assert parameters.random_state == 42
    assert parameters.test_size == 0.2
    assert parameters.gridsearch_scoring == "neg_mean_squared_error"
    assert parameters.gridsearch_n_splits == 18
    assert parameters.model_cpu_params == configurator.MODEL_CPU_PARAMS
    assert parameters.model_params['objective'] == ["reg:squarederror"]
    assert parameters.model_params['n_estimators'] == [50, 100, 300]
    assert parameters.model_params['learning_rate'] == [0.005, 0.05, 0.1, 0.2]
    assert parameters.model_params['max_depth'] == [3, 5, 8, 15]


def test_custom_configuration():
    """
    Test for getting custom configuration
    """
    custom_config = {
        "use_gridsearch": False,
        "random_state": 43,
        "test_size": 0.6,
        "gridsearch_scoring": "mean_squared_error",
        "gridsearch_n_splits": 6,
        "dataset_cleaning_params": {
            "col_max_na_percentage": 100,
            "row_max_na_percentage": 100
        },
        "model_cpu_params": {
            "objective": "reg:squarederror",
            "n_estimators": 81,
            "learning_rate": 0.9,
            "n_jobs": 1,
            "predictor": "cpu_predictor",
            "tree_method": "auto",
            "max_depth": 10
        },
        "model_params": {
            "objective": "reg:squarederror",
            "n_estimators": 81,
            "learning_rate": 0.3,
            "n_jobs": -1,
            "max_depth": 9,
            "tree_method": "approx",
            "predictor": "cpu_predictor"
        }
    }
    mock_open = mock.mock_open(read_data=json.dumps(custom_config))
    with mock.patch('builtins.open', mock_open):
        configurator = CrossCorrelationConfigurator(
            xcorr_configuration_file="/dev/null")
        parameters = configurator.get_configuration()

        assert parameters.use_gridsearch == custom_config['use_gridsearch']
        assert parameters.random_state == custom_config['random_state']
        assert parameters.test_size == custom_config['test_size']
        assert parameters.gridsearch_scoring == custom_config[
            'gridsearch_scoring']
        assert parameters.gridsearch_n_splits == custom_config[
            'gridsearch_n_splits']
        assert parameters.model_cpu_params == custom_config['model_cpu_params']
        assert parameters.model_params == custom_config['model_params']
