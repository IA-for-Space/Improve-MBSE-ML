"""
`pytest` testing framework file for detector configurator
"""

import json
from unittest import mock

from polaris.anomaly.anomaly_detector_configurator import \
    AnomalyDetectorConfigurator
from polaris.feature.cleaner_configurator import CleanerConfigurator


def test_get_default_configuration():
    """
    Test for getting default configuration
    """
    configurator = AnomalyDetectorConfigurator()
    parameters = configurator.get_configuration()

    assert parameters.window_size == 2
    assert parameters.stride == 1
    assert parameters.optimizer == "adam"
    assert parameters.loss == "mean_squared_error"
    assert parameters.metrics == ["MSE"]
    assert parameters.test_size_fraction == 0.2
    assert parameters.number_of_epochs == 20
    assert parameters.noise_margin_per == 50
    assert parameters.batch_size == 128
    assert parameters.network_dimensions == [64, 32]
    assert parameters.activations is None


def test_custom_configuration():
    """
    Test for getting custom configuration
    """
    custom_config = {
        "window_size": 3,
        "stride": 2,
        "optimizer": "rmsprop",
        "loss": "mean_squared_error",
        "metrics": ["MSE"],
        "test_size_fraction": 0.4,
        "number_of_epochs": 40,
        "noise_margin_per": 150,
        "batch_size": 64,
        "network_dimensions": [128, 64, 32],
        "activations": ["relu"],
        "dataset_cleaning_params": {
            "col_max_na_percentage": 100,
            "row_max_na_percentage": 100
        },
    }
    mock_open = mock.mock_open(read_data=json.dumps(custom_config))
    with mock.patch('builtins.open', mock_open):
        configurator = AnomalyDetectorConfigurator(
            detector_configuration_file="/dev/null")
        parameters = configurator.get_configuration()
        for props in custom_config:
            if props == "dataset_cleaning_params":
                cleaner_config = CleanerConfigurator(custom_config[props])
                cleaner_params = cleaner_config.get_configuration()
                output_cleaner_params = parameters.dataset_cleaning_params

                assert output_cleaner_params.col_max_na_percentage == \
                    cleaner_params.col_max_na_percentage
                assert output_cleaner_params.row_max_na_percentage == \
                    cleaner_params.row_max_na_percentage
            else:
                assert getattr(parameters, props) == custom_config[props]
