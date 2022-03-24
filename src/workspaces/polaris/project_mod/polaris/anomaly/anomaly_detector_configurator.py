"""Module for AnomalyDetectorConfigurator class
"""

import json
import logging
import warnings

from polaris.anomaly.anomaly_detector_parameters import \
    AnomalyDetectorParameters
from polaris.feature.cleaner_configurator import CleanerConfigurator

LOGGER = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
class AnomalyDetectorConfigurator():
    """Class for configurator for model used in detector
    """
    def __init__(self, detector_configuration_file=None):
        """ Initialize model configuration

            :param detector_configuration_file: configuration file path,
                defaults to None
            :type detector_configuration_file: str, optional
        """
        self._detector_configuration_file = detector_configuration_file
        self._anomaly_detector_parameters = AnomalyDetectorParameters()
        super().__init__()

    def get_configuration(self):
        """ Turn Detector configuration file to Detector parameters

            :return: If there is an Detector configuration file, the parameters
                for the Detector configuration are returned. Otherwise, the
                default Detector parameters are returned
        """

        if self._detector_configuration_file is not None:
            self._get_configuration_from_file(
                self._detector_configuration_file)
            return self._anomaly_detector_parameters

        self._set_default_detector_parameters()
        return self._anomaly_detector_parameters

    def _set_default_detector_parameters(self):
        """Set default detector parameters if no configurator file is found
        """
        self._anomaly_detector_parameters.window_size = 2
        self._anomaly_detector_parameters.stride = 1
        self._anomaly_detector_parameters.optimizer = "adam"
        self._anomaly_detector_parameters.loss = "mean_squared_error"
        self._anomaly_detector_parameters.metrics = ["MSE"]
        self._anomaly_detector_parameters.test_size_fraction = 0.2
        self._anomaly_detector_parameters.number_of_epochs = 20
        self._anomaly_detector_parameters.noise_margin_per = 50
        self._anomaly_detector_parameters.batch_size = 128
        self._anomaly_detector_parameters.network_dimensions = [64, 32]
        self._anomaly_detector_parameters.activations = None

        feature_cleaner = CleanerConfigurator()
        self._anomaly_detector_parameters.dataset_cleaning_params = \
            feature_cleaner.get_configuration()

    def _get_configuration_from_file(self, path):
        """ Read custom config from file.

            :raises Exception: If the custom configuration file
                failed to load
        """
        LOGGER.info("Using custom configuration!")
        try:
            with open(path, "r") as config_file:
                config_data = json.load(config_file)

            self._set_custom_configuration(**config_data)
        except Exception as exception_error:
            LOGGER.critical(exception_error)
            raise exception_error

    # pylint: disable=too-many-arguments
    def _set_custom_configuration(self, window_size, stride, optimizer, loss,
                                  metrics, test_size_fraction,
                                  number_of_epochs, noise_margin_per,
                                  batch_size, network_dimensions, activations,
                                  dataset_cleaning_params):
        """ Set all the detector_parameters properties.

            :param window_size: window size for preprocessing of Data
            :type window_size: int
            :param stride: Stride count for preprocessing of Data
            :param stride: int
            :param optimizer: type of optimizer for compiling model
            :type optimizer: string
            :param loss: type of loss metric for compiling model
            :type loss: string
            :param metrics: metrics to measure loss in compiling model
            :type metrics: dict[string]
            :param test_size_fraction: fraction of test compared to whole
            dataset, ranging from 0 to 1
            :type test_size_fraction: float
            :param number_of_epochs: number of epochs to run
            :type number_of_epochs: int
            :param noise_margin_per: percentage of noise above which
                anomaly is to be detected
            :type noise_margin_per: int
            :param batch_size: batch size to be processed
            :type batch_size: int
            :raises TypeError: If metrics is not a Python dictionary
                or if there is one value in metrics that is not a
                Python list
        """
        self._anomaly_detector_parameters.window_size = window_size
        self._anomaly_detector_parameters.stride = stride
        self._anomaly_detector_parameters.optimizer = optimizer
        self._anomaly_detector_parameters.loss = loss
        self._anomaly_detector_parameters.test_size_fraction = \
            test_size_fraction
        self._anomaly_detector_parameters.number_of_epochs = number_of_epochs
        self._anomaly_detector_parameters.noise_margin_per = noise_margin_per
        self._anomaly_detector_parameters.batch_size = batch_size
        self._anomaly_detector_parameters.activations = activations

        feature_cleaner = CleanerConfigurator(dataset_cleaning_params)
        self._anomaly_detector_parameters.dataset_cleaning_params = \
            feature_cleaner.get_configuration()

        if not isinstance(metrics, list):
            raise TypeError("Expected {} got {}".format(list, type(metrics)))
        self._anomaly_detector_parameters.metrics = metrics
        if not isinstance(network_dimensions, list):
            raise TypeError("Expected {} got {}".format(
                list, type(network_dimensions)))
        self._anomaly_detector_parameters.network_dimensions = \
            network_dimensions
