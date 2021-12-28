"""Test for polaris.anomaly.detector
"""

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from polaris.anomaly.anomaly_detector import AnomalyDetector
from polaris.anomaly.anomaly_detector_configurator import \
    AnomalyDetectorConfigurator
from polaris.data.readers import read_polaris_data

SAMPLE_DATA = pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": [11, 15, 19, 23, 27],
    "c": [50, 100, 150, 200, float("nan")],
    "d": [.5, 1.5, 3.5, 12.5, 15.5],
})

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "test_data")


@contextmanager
def does_not_raise():
    """Dummy class for cases where errors are not raise
    """
    yield


def get_empty_detector():
    """Give instance of an empty detector without any data
    """
    configurator = AnomalyDetectorConfigurator()
    parameters = configurator.get_configuration()
    metadata = {}
    detector = AnomalyDetector(dataset_metadata=metadata,
                               anomaly_detector_params=parameters)
    return detector


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "normalized_frames.json"))
def test_train_predict_output(datafiles):
    """
    `pytest` entry point
    """
    metadata, data = read_polaris_data(
        str(datafiles / "normalized_frames.json"))

    configurator = AnomalyDetectorConfigurator()
    parameters = configurator.get_configuration()

    detector = AnomalyDetector(dataset_metadata=metadata,
                               anomaly_detector_params=parameters)
    detector.train_predict_output(data)


def test_create_compile_models():
    """Test for creating models
    """
    detector = get_empty_detector()
    detector.create_models()
    detector.compile_models()


def test_detect_events():
    """Test for detecting events
    """
    detector = get_empty_detector()
    detector.anomaly_detector_params.noise_margin_per = 0
    df_pred_bn = np.array([
        [0.9, 1.5, 30.4, 2.5, 12],
        [13, 2.5, 33.9, 1.5, 10],
        [14, 1.3, 28.7, 3.5, 9],
        [1.42, 1.3, 26.3, 2.3, 13],
        [0.9, 1.5, 30.4, 2.5, 12],
        [13, 2.5, 33.9, 1.5, 10],
    ])
    events = detector.detect_events(df_pred_bin=df_pred_bn)
    assert len(events) >= 1
