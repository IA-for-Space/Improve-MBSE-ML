"""
Module for fetching and decoding space weather data
"""
import datetime
import importlib
import logging

import pandas as pd
from dateutil import parser
from vinvelivaanilai.space_weather import sw_file_fetch
from vinvelivaanilai.space_weather.sw_extractor import NoSpaceWeatherForIndex
from vinvelivaanilai.storage import retrieve, store

from polaris.fetch import fetch_import_telemetry

NORMALIZER_LIB = 'contrib.normalizers.'

SUPPORTED_INDICES = ('DGD', 'DPD', 'DSD')

LOGGER = logging.getLogger(__name__)


class InfluxDBError(Exception):
    """
    Class to raise errors from InfluxDB
    """


def load_sw_normalizer(index):
    """
    Load normalizer for space weather indices

    :param index: Name of the index for which normalizer needs to be loaded
    :type index: str
    :return: Loaded normalizer
    """
    try:
        LOGGER.info("Loading normalizer for index: %s", index)
        loaded_normalizer = importlib.import_module(NORMALIZER_LIB +
                                                    "space_weather")
        normalizer = getattr(loaded_normalizer, index.upper())
        return normalizer
    except Exception as eee:
        LOGGER.error("Normalizer loading: %s", eee)
        raise eee


def fetch_sw(start_date, end_date, cache_dir, indices=SUPPORTED_INDICES):
    """
    Fetch Space Weather indices using vinvelivaanilai.

    :param start_date: Start date of space weather data to fetch
    :type start_date: datetime.datetime
    :param end_date: End date of space weather data to fetch
    :type end_date: datetime.datetime
    :param cache_dir: Cache directory where downloaded files are stored
    :type cache_dir: str
    :param indices: List of indices to fetch, defaults to SUPPORTED_INDICES
    :type indices: list, optional
    :return: Dictionary of dataframes containing indices fetched
    :rtype: dict of pd.DataFrame
    """
    data = {}
    for index in indices:
        if index not in SUPPORTED_INDICES:
            raise ValueError("Index {} not supported yet!".format(index))

        try:
            temp_df = sw_file_fetch.fetch_indices(index, start_date, end_date,
                                                  cache_dir)
            # To prevent problems for learn
            data[index] = temp_df.fillna(-1)
        except NoSpaceWeatherForIndex as err:
            LOGGER.info(err)
            continue

    return data


def fetch_sw_from_influxdb(sat,
                           local_start_date,
                           local_end_date,
                           indices=SUPPORTED_INDICES):
    """
    Fetch the space weather data from influxdb

    :param sat: Name of satellite (Doubles as name of bucket)
    :type sat: str
    :param local_start_date: Start date of space weather data to fetch
    :type local_start_date: datetime.datetime
    :param local_end_date: End date of space weather data to fetch
    :type local_end_date: datetime.datetime
    :param indices: List of indices to fetch, doubles as measurement_name
        defaults to SUPPORTED_INDICES
    :type indices: list, optional
    :return: Dictionary of dataframes containing indices fetched
    :rtype: dict of pd.DataFrame
    """
    # Fetch data from influxdb
    LOGGER.info("Fetching space weather data from influxdb")
    data = {}
    for index in indices:
        try:
            data[index] = retrieve.fetch_from_influxdb(
                datetime.datetime(year=local_start_date.year,
                                  month=local_start_date.month,
                                  day=local_start_date.day),
                datetime.datetime(year=local_end_date.year,
                                  month=local_end_date.month,
                                  day=local_end_date.day),
                index,
                sat,
                rename_to="Date")

            # There may be no data for that period
            if data[index] is None:
                LOGGER.info("No data found for %s in the given interval",
                            index)
                del data[index]

        except Exception:
            LOGGER.error(
                "Fetching space weather data from influxdb failed. "
                "Possible reasons are:\n"
                "1. Indluxdb did not start properly\n"
                "2. Incorrect credentials\n"
                "3. There was no data for specified time range (Did you "
                "run fetch for this time range before?)")
            raise InfluxDBError
    return data


def store_sw(data, measurement_name, bucket_name):
    """
    Store space weather data in influxdb

    :param data: Dataframe to store
    :type data: pd.DataFrame
    :param measurement_name: Name of the measurement to store under in idb
    :type measurement_name: str
    :param bucket_name: Name of the bucket to store under in idb
    :type bucket_name: str
    """
    LOGGER.debug("Storing %s data in Influxdb", measurement_name)
    try:
        store.dump_to_influxdb(data, measurement_name, bucket_name)
    except Exception:
        LOGGER.error(
            "Error storing data in influxdb. Are you sure the credentials"
            " are correct?")
        raise InfluxDBError


def fetch_or_import_sw(start_date, end_date, cache_dir, sat, **kwargs):
    """
    Fetch/import space weather from vinvelivaanilai/txt/influxdb.

    :param start_date: Start date of data to fetch
    :type start_date: str
    :param end_date: End date of data to fetch
    :type end_date: str
    :param cache_dir: Path to cache directory
    :type cache_dir: str
    :param sat: Name of the satellite
    :type sat: str
    :param **kwargs: See below. If kwargs are not passed, it will fetch from
        vinvelivaanilai

    :Keyword Arguments:
        * store_in_influxdb (bool) -- True if influxdb is to be used. Default
            False.
        * fetch_from_influxdb (bool) -- True if data is fetched from influxdb.
            Default False.
        * indices (list) -- List of indices to fetch. Default SUPPORTED_INDICES

    :return: Dictionary with index as space_weather indices and values as
        corresponding pd.DataFrame
    :rtype: dict of pd.DataFrame
    """
    # Get all valid kwargs
    store_in_influxdb = kwargs.pop("store_in_influxdb", False)
    fetch_from_influxdb = kwargs.pop("fetch_from_influxdb", False)
    indices = kwargs.pop("indices", SUPPORTED_INDICES)

    # Get the times as datetime objects
    local_start_date = parser.parse(start_date)
    local_end_date = parser.parse(end_date)
    if fetch_from_influxdb:
        try:
            data = fetch_sw_from_influxdb(sat, local_start_date,
                                          local_end_date, indices)
            return data
        except InfluxDBError:
            LOGGER.error(
                "Error fetching from influxdb, fetching from vinvelivaanilai")

    # Download the data
    LOGGER.info("Fetching space weather data from vinvelivaanilai")
    data = fetch_sw(local_start_date, local_end_date, cache_dir, indices)
    if store_in_influxdb:
        try:
            for index in data:
                store_sw(data[index].copy(), index, sat)
        except InfluxDBError:
            LOGGER.error("Error storing data in influxdb!"
                         " Proceeding WITHOUT storing data")

    return data


def fetch_nearest_sw(sw_data, time_list):
    """
    Find the rows of space weather data nearest to each entry in time_list

    :param sw_data: Dictionary of dataframes with key as index
    :type sw_data: dict
    :param time_list: List of times for which we need the nearest data
    :type time_list: list
    :return: Dictionary of dataframes corresponding to each index after finding
        rows nearest to times in time_list
    :rtype: dict
    """
    LOGGER.info("Finding the nearest space weather data")
    # Convert the list of times to pd.DatetimeIndex
    time_df = pd.to_datetime(time_list,
                             infer_datetime_format=True).sort_values()
    processed_data = {}
    for index in sw_data:
        # Fetch the nearest data for each index
        processed_data[index] = retrieve.get_multiple_nearest_from_df(
            time_df.astype(sw_data[index].index.dtype), sw_data[index])

    return processed_data


def dataframe_to_decoded(dataframe):
    """
    Convert pd.DataFrame to the decoded_frames format

    :param dataframe: DataFrame to convert
    :type dataframe: pd.DataFrame
    :return: List of frames in decoded_frames format
    :rtype: list
    """
    local_df = dataframe.copy().reset_index().T
    data_dict = local_df.to_dict()

    decoded_data = []
    for value in data_dict.values():
        decoded_data.append({
            "time": value['Date'],
            "fields": {
                key: value[key]
                for key in set(list(value.keys())) - set(['Date'])
            }
        })
    return decoded_data


def fetch_preprocessed_sw(start_date, end_date, cache_dir, time_list, sat,
                          **kwargs):
    """
    Fetch and preprocess space weather data

    :param start_date: Start date of data to fetch
    :type start_date: str
    :param end_date: End date of data to fetch
    :type end_date: str
    :param cache_dir: Where temporary output data should go
    :type cache_dir: str, os.path
    :param time_list: List of times for which nearest space weather needs to
        be fetched
    :type time_list: list
    :param sat: Name of the satellite
    :type sat: str

    :return: Columns names and dictionary with keys as the indices and values
        as the preprocessed frames of space_weather
    :rtype: (list, dict)
    """
    if start_date is None:
        start_date = min(time_list)

    if end_date is None:
        end_date = max(time_list)

    # Fetch the space_weather data
    sw_data = fetch_or_import_sw(start_date, end_date, cache_dir, sat,
                                 **kwargs)

    # Get the nearest data
    nearest_sw_data = fetch_nearest_sw(sw_data, time_list)

    LOGGER.info(
        "Converting space weather data to the normalized frames format")
    # Preprocess it into the same format as normalized telemetry frames
    preprocessed_sw_frames = {}
    columns_names = []
    for index in nearest_sw_data:
        decoded_sw_frame = dataframe_to_decoded(nearest_sw_data[index])
        sw_normalizer = load_sw_normalizer(index)()
        columns_names.extend(sw_normalizer.get_fields_name())
        preprocessed_sw_frames[index] = fetch_import_telemetry.data_normalize(
            sw_normalizer, decoded_sw_frame)

    return columns_names, preprocessed_sw_frames
