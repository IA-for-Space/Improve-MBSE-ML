"""
Anomaly Output class
It is used for output the result of AnomalyDetector in json
"""

import json

import pandas as pd
from betsi.preprocessors import convert_from_column

from polaris.anomaly.anomaly_detector import AnomalyDetector
from polaris.common import constants
from polaris.common.json_serializable import JsonSerializable
from polaris.dataset.metadata import PolarisMetadata


class AnomalyOutput(dict, JsonSerializable):
    """Class for Output the result of AnomalyDetector
    """
    def __init__(self, metadata=None):
        """Initialize a new object
        """
        dict.__init__(self)
        JsonSerializable.__init__(self)
        self.metadata = PolarisMetadata(metadata)
        self.data = {"timestamps": None, "events": None, "values": {}}

    def from_detector(self, detector: AnomalyDetector):
        """
        Function to set output from detector given

        param detector: detector from which output will be made
        type detector: AnomalyDetector
        """
        time_index = detector.time_index.to_list()
        time_index = [str(time) for time in time_index]
        self.data["timestamps"] = time_index

        original_data = self.get_original_data(detector)

        events = detector.events
        values = {}
        for col in original_data.columns:
            col_values = original_data[col].to_list()
            col_events = events[col]
            updated_events = [time_index[x - 1] for x in col_events]
            values[col] = {
                "individual_values": col_values,
                "individual_events_detected": updated_events
            }

        self.data = {
            "timestamps": time_index,
            "events": events["overall"],
            "values": values
        }

    @staticmethod
    def get_original_data(detector: AnomalyDetector):
        """
        Function to get original data from preprocessed data.

        :param detector: Detector used to detect the events
        """
        window_size = detector.anomaly_detector_params.window_size
        stride = detector.anomaly_detector_params.stride

        converted_data = convert_from_column(detector.preprocessed_data,
                                             window_size, stride)

        # it is because convert_from_columns does not completely return
        # original data. When convert_from_column method from preprocessors
        # of Betsi is updated, there will be no need of it.
        # Related issue:
        # https://gitlab.com/librespacefoundation/polaris/betsi/-/issues/18
        original_data = pd.DataFrame(converted_data,
                                     columns=converted_data.columns)
        columns = converted_data.columns
        columns = [col[:-1] for col in columns]
        original_data.columns = columns
        return original_data

    def show(self):
        """ Get dictionary representation to represent
        """
        return {"metadata": self.metadata, "data": self.data}

    def __repr__(self):
        return repr(self.to_json())

    def __str__(self):
        return json.dumps(self.to_json(), indent=constants.JSON_INDENT)

    def to_json(self):
        """Write a dataset object to JSON.
        """
        return json.dumps(self.show(), indent=constants.JSON_INDENT)
