"""
Module to launch anomaly data analysis.
"""
import logging
import os

from mlflow import set_experiment, start_run, tensorflow

from polaris.anomaly.anomaly_detector import AnomalyDetector
from polaris.anomaly.anomaly_detector_configurator import \
    AnomalyDetectorConfigurator
from polaris.anomaly.anomaly_output import AnomalyOutput
from polaris.common.util import create_parent_directory
from polaris.data.readers import read_polaris_data
from polaris.learn.analysis import NoFramesInInputFile

LOGGER = logging.getLogger(__name__)


class FileIsADirectory(Exception):
    """Raised when the file path is of a directory
    """


# pylint: disable-msg=too-many-arguments
def behave(input_file,
           output_file="/tmp/anomaly_output.json",
           detector_config_file=None,
           cache_dir='/tmp',
           metrics_dir='/tmp',
           csv_sep=',',
           save_test_train_data=False):
    """
    Detect events in input data and output anomaly events

        :param input_file: CSV or JSON file path that will be
            converted to a dataframe
        :type input_file: str

        :param output_file: Output file path for the generated graph.
            It will overwrite if the file already exists.
        :type output_file: str, optional

        :param detector_config_file: Detector configuration file path,
            defaults to None. Refer to CrossCorrelationConfigurator for
            the default parameters
        :type detector_config_file: str, optional

        :param cache_dir: directory to store files such as models used,
            normalizer, training data
        :type cache_dir: str

        :param metrics_dir: directory to store anomaly metrics
        :type metrics_dir: str

        :param csv_sep: The character that separates the columns inside of
            the CSV file, defaults to ','
        :type csv_sep: str, optional

        :param save_test_train_data: decides weather to save test and
            train data in cache or not
        :type save_test_train_data: Boolean

        :raises NoFramesInInputFile: If there are no frames in the converted
            dataframe
    """
    if os.path.isdir(output_file):
        LOGGER.error("output file path is a directory")
        raise FileIsADirectory

    metadata, dataframe = read_polaris_data(input_file, csv_sep)

    if dataframe.empty:
        LOGGER.error("Empty list of frames -- nothing to learn from!")
        raise NoFramesInInputFile

    set_experiment(experiment_name=metadata['satellite_name'])

    #  getting parameters for anomaly detector
    anomaly_config = AnomalyDetectorConfigurator(
        detector_configuration_file=detector_config_file)
    anomaly_params = anomaly_config.get_configuration()

    tensorflow.autolog()
    # creating detector and detecting events
    detector = AnomalyDetector(dataset_metadata=metadata,
                               anomaly_detector_params=anomaly_params)
    with start_run(run_name="behave analysis"):
        anomaly_metrics = detector.train_predict_output(data=dataframe)

    # saving data generated by detector
    detector.save_artifacts(cache_dir, save_test_train_data)

    detector.save_anomaly_metrics(metrics_dir, anomaly_metrics)

    output = AnomalyOutput(metadata=metadata)
    output.from_detector(detector=detector)

    create_parent_directory(output_file)
    with open(output_file, 'w') as graph_file:
        graph_file.write(output.to_json())