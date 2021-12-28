"""
`pytest` testing framework file for AnomalyDetectorParameters
"""

from polaris.anomaly.anomaly_detector_parameters import \
    AnomalyDetectorParameters


def test_getter_setter():
    """
    Test for getters and setters of class.
    """
    values_to_set = {
        "window_size": 10,
        "stride": 5,
        "optimizer": "adam",
    }
    params = AnomalyDetectorParameters()
    for key, value in values_to_set.items():
        setattr(params, key, value)

    for key, value in values_to_set.items():
        assert getattr(params, key) == value
