"""
Anomaly Detector module
"""

import json
import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from betsi.models import custom_autoencoder
from betsi.predictors import distance_measure, get_events
from betsi.preprocessors import convert_from_column, convert_to_column, \
    normalize_all_data
from mlflow import log_metric, log_param
from sklearn.model_selection import train_test_split

from polaris.anomaly.anomaly_detector_parameters import \
    AnomalyDetectorParameters
from polaris.feature.cleaner import Cleaner

LOGGER = logging.getLogger(__name__)


class AnomalyDetector():
    """ Anomaly Detector class
    """
    def __init__(self, dataset_metadata,
                 anomaly_detector_params: AnomalyDetectorParameters):
        """ Initialize an AnomalyDetector object
            :param dataset_metadata: metadata of the satellite
            :type dataset_metadata: PolarisMetadata
            :param anomaly_detector_params: AnomalyDetector parameters
            :type anomaly_detector_params: AnomalyDetectorParameters
        """
        self.models = None
        self.normalizer = None
        self.preprocessed_data = None
        self.events = None
        self.time_index = None

        self.anomaly_detector_params = anomaly_detector_params
        self._feature_cleaner = Cleaner(
            dataset_metadata, anomaly_detector_params.dataset_cleaning_params)

    @staticmethod
    def timeseries_sort_by_timestamp(data):
        """
        Function to sort the data according to timestamps indexes
        :param data: DataFrame to sort
        :type data: pd.DataFrame
        :return: sorted data
        :rtype: pd.DataFrame
        """
        sorted_data = data.sort_values(by="time")
        sorted_data.reset_index(inplace=True)
        del sorted_data['index']
        return sorted_data

    def clean_data(self, data):
        """
        Function to clean data like
        drop non numeric values and handling missing values

        :param data: DataFrame which will undergo clean
        :type data: pd.DataFrame
        :return: cleaned data
        :rtype: pd.DataFrame
        """
        feature_cleaner = self._feature_cleaner
        data = feature_cleaner.drop_non_numeric_values(data)
        data = feature_cleaner.handle_missing_values(data)
        data = data.select_dtypes("number")
        return data

    def get_train_test_data(self):
        """
        Function to split data into train and test set
        return: tuple containing train and test data
        rtype: tuple
        """
        test_size = self.anomaly_detector_params.test_size_fraction
        # Split it into train and test data
        # set shuffle to false as order matters
        # since it is time series data
        preprocessed_data = self.preprocessed_data
        train_data, test_data = train_test_split(preprocessed_data,
                                                 test_size=test_size,
                                                 shuffle=False)
        data = (train_data, test_data)
        return data

    def create_models(self):
        """Creates the 3 models: autoencoder, encoder and decoder

        :raises ValueError: If the model could not be created
        """
        layer_dims = self.anomaly_detector_params.network_dimensions
        activations = self.anomaly_detector_params.activations
        try:
            self.models = custom_autoencoder(layer_dims, activations)

        except ValueError as err:
            LOGGER.error("Error creating the model. This might be because:")
            LOGGER.error("1. Layer dimensions are incorrect")
            LOGGER.error("2. Activation specified does not exist")
            LOGGER.error(err)
            raise err

    def compile_models(self):
        """
        Function to create and compile the models
        """

        optimizer = self.anomaly_detector_params.optimizer
        loss = self.anomaly_detector_params.loss
        metrics = self.anomaly_detector_params.metrics

        autoencoder_model, encoder_model, decoder_model = self.models

        # Compiling the autoencoder model
        autoencoder_model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=metrics)
        encoder_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        decoder_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.models = (autoencoder_model, encoder_model, decoder_model)

    def train_test_model(self):
        """Function to train and test the compiled model

        :return: training history
        :rtype: dict
        """

        batch_size = self.anomaly_detector_params.batch_size
        epochs = self.anomaly_detector_params.number_of_epochs

        (autoencoder_model, _, _) = self.models
        train_data, test_data = self.get_train_test_data()

        # Train the model for epochs number of epochs, keep history for
        # further analysis
        LOGGER.info(
            "Training on %i rows of data, with batch size %i and %i epochs",
            train_data.shape[0], batch_size, epochs)
        try:
            history = autoencoder_model.fit(train_data,
                                            train_data,
                                            batch_size=batch_size,
                                            epochs=epochs)
        except Exception as err:
            # Exception if data not formatted properly, gradients vanishing
            # or some model errors
            LOGGER.critical("Error fitting data. Aborting anomaly detection")
            LOGGER.critical(err)
            raise err

        # Evaluate the model on test data and log results.
        # It's up to the user to decide if it has overfitted.
        test_results = autoencoder_model.evaluate(test_data,
                                                  test_data,
                                                  batch_size=batch_size)
        log_metric("Test loss", test_results[0])
        log_metric("Test MSE", test_results[1])
        LOGGER.info("Test loss: %s, Test MSE: %s", str(test_results[0]),
                    str(test_results[1]))
        return history

    def save_artifacts(self, cache_dir, save_test_train_data=False):
        """Save important artifacts like normalizers,
            models and test/train data

        :param save_test_train_data: weather to save test train data
        :param cache_dir: Path to cache directory
        """
        if not os.path.isdir(cache_dir):
            try:
                os.makedirs(cache_dir)
            except Exception as err:
                LOGGER.critical(
                    "Error creating the path %s."
                    "Do you have the correct permissions?", str(cache_dir))
                LOGGER.critical(err)
                raise err

        autoencoder_model, encoder_model, decoder_model = self.models
        normalizer = self.normalizer
        # Save models to respective files
        autoencoder_model.save(
            os.path.join(cache_dir, "autoencoder_model.tf_model"))
        encoder_model.save(os.path.join(cache_dir, "encoder_model.tf_model"))
        decoder_model.save(os.path.join(cache_dir, "decoder_model.tf_model"))

        if save_test_train_data:
            tt_data = self.get_train_test_data()
            # Save test and train data for future use (while visualizing)
            tt_data[0].to_pickle(os.path.join(cache_dir, "train_data.pkl"))
            tt_data[1].to_pickle(os.path.join(cache_dir, "test_data.pkl"))

        # Save the normalizer to preprocess data next time
        joblib.dump(normalizer, os.path.join(cache_dir, "normalizer.pkl"))

    @staticmethod
    def save_anomaly_metrics(cache_dir, anomaly_metrics):
        """Save anomaly_metrics in a single file

        :param cache_dir: Path to cache directory
        :param anomaly_metrics: Dictionary containing other metrics
        """

        # Make directory if not exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # Save all the anomaly metrics (training history, events)
        with open(os.path.join(cache_dir, "anomaly_metrics.json"),
                  "w") as json_file:
            json.dump(anomaly_metrics, json_file)

    def detect_events(self, df_pred_bin=None):
        """
        Function to detect anomalies/events

        :param df_pred_bin: compact representation of input from AE model
        :type df_pred_bin: pd.DataFrame, optional
        :return: list of event indices
        :rtype: list
        """
        noise_margin_per = self.anomaly_detector_params.noise_margin_per
        feature_data = df_pred_bin

        if feature_data is None:
            data = self.preprocessed_data
            encoder_model = self.models[1]
            feature_data = encoder_model.predict(data)

        # Get distances
        distance_list = []
        for row_no in range(feature_data.shape[0] - 1):
            distance_list.append(
                distance_measure(feature_data[row_no],
                                 feature_data[row_no + 1]))

        events = get_events(distance_list, threshold=noise_margin_per)
        return events

    def detect_individual_events(self):
        """
        Function to detect events for individual parameters of data
        """

        window_size = self.anomaly_detector_params.window_size
        stride = self.anomaly_detector_params.stride
        noise_margin_per = self.anomaly_detector_params.noise_margin_per

        # to get data of individual columns
        data = convert_from_column(self.preprocessed_data, window_size, stride)

        # it is because convert_from_columns doesnot completely return original
        # data. when the betsi will be updated, there will be no need of it
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            columns = [col[:-1] for col in columns]
            data.columns = columns

        res = {}
        for col in data.columns:
            col_data = data[col].tolist()
            distance_list = []

            # generating the distance list
            for row_no in range(len(col_data) - 1):
                curr_item = abs(col_data[row_no])
                next_item = abs(col_data[row_no + 1])
                if (curr_item == 0 or next_item == 0):
                    value_to_append = 0
                else:
                    value_to_append = (curr_item - next_item) / math.sqrt(
                        curr_item * next_item)
                distance_list.append(value_to_append)

            # detecting events with distance list
            events = get_events(distance_list, threshold=noise_margin_per)
            res[col] = events
        res["overall"] = self.detect_events()
        self.events = res

    def set_data(self, data):
        """
        Function to clean , normalize and convert to columns
        the input data set properties like
        preprocessed_data, time_index of the detector

        :param data: Input Data
        :type data: DataFrame
        """
        # From config file
        layer_dims = self.anomaly_detector_params.network_dimensions
        window_size = self.anomaly_detector_params.window_size
        stride = self.anomaly_detector_params.stride
        feature_cleaner = self._feature_cleaner

        #  save time index for future reference
        self.time_index = pd.to_datetime(data.time, unit="s")

        # necessary droping constant values before normalizing
        data = feature_cleaner.drop_constant_values(data)

        # Normalize the data
        normalized_data, normalizer = normalize_all_data(data)
        self.normalizer = normalizer

        # clean data
        cleaned_data = self.clean_data(normalized_data)

        # convert data into columns
        converted_data = convert_to_column(cleaned_data, window_size, stride)

        self.preprocessed_data = converted_data

        self.anomaly_detector_params.network_dimensions = [
            converted_data.shape[1]
        ] + layer_dims

    def train_predict_output(self, data):
        """
        Train data, predict events and output metrics

        :return: anomaly_metrics containing history of training Data
        :rtype: dict
        """
        self.mlf_params_logging()
        sorted_data = self.timeseries_sort_by_timestamp(data)
        self.set_data(sorted_data)

        #  create and compile models
        self.create_models()
        self.compile_models()

        #  train and test model
        history = self.train_test_model()

        # detect events
        self.detect_individual_events()

        anomaly_metrics = {
            "history": history.history,
        }
        return anomaly_metrics

    def mlf_params_logging(self):
        """ Log the test_size and Model in mlflow
        """
        log_param('Test size', self.anomaly_detector_params.test_size_fraction)
        log_param('Model', 'XGBRegressor')
