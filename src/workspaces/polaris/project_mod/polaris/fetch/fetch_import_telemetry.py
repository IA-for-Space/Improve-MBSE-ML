"""
Module to fetch, decode and normalize satellite telemetry data
"""

import datetime
import glob
import importlib
import json
import logging
import os
import subprocess
import sys

import pandas as pd
# import glouton dependencies
from glouton.domain.parameters.programCmd import ProgramCmd
from glouton.services.observation.observationsService import \
    ObservationsService

NORMALIZER_LIB = 'contrib.normalizers.'

LOGGER = logging.getLogger(__name__)


def build_decode_cmd(src, dest, decoder):
    """ Build command to decode downloaded into JSON """
    decode_multiple = 'decode_multiple'
    decoder_module = decoder
    input_format = 'csv'
    decode_cmd = '{decode_multiple} --filename {src} --format {input_format}'\
                 ' {decoder_module} > {dest}'.format(
                     decode_multiple=decode_multiple,
                     decoder_module=decoder_module,
                     src=src,
                     input_format=input_format,
                     dest=dest,
                 )
    return decode_cmd  # pylint: disable=R0914


class NoNormalizerForSatellite(Exception):
    """Raised when we have no normalizer """


class NoCSVFilesToMerge(Exception):
    """Raised when there are no CSV files to merge or the downloaded
    CSV files have been modified"""


class NoDecodedFramesFile(Exception):
    """Raised when there is no file of decoded frames after attempting to
    download new frames.
    """


class DecodeMultipleFailed(Exception):
    """Raised when decode_multiple command fails"""


class SpecifiedImportFileDoesNotExist(Exception):
    """Raised when a specified file to be imported does not exist.
    """


def load_normalizer(sat):
    """
    Load the normalizer selected by name within the NORMALIZER_LIB.

    :param sat: a satellite object.

    :returns: the loaded normalizer.
    """
    if sat == "Dummy":
        normalizer_name = sat
    elif sat.normalizer is None:
        raise NoNormalizerForSatellite
    else:
        normalizer_name = sat.normalizer

    try:
        LOGGER.debug("Normalizer chosen: %s", normalizer_name)
        loaded_normalizer = importlib.import_module(NORMALIZER_LIB +
                                                    normalizer_name.lower())
        normalizer = getattr(loaded_normalizer, normalizer_name)
        return normalizer
    except Exception as eee:
        LOGGER.error("Error loading normalizer: %s", normalizer_name)
        raise eee


def build_start_and_end_dates(start_date, end_date):
    """
    Build start and end dates using either provided string, provided
    datetime object, or choosing default.

    Default period starts one hour before UTC current time at call.

    """
    # First start date; if no date provided, set to an hour ago.
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    elif not isinstance(start_date, datetime.datetime):
        start_date = (pd.Timestamp.utcnow() - pd.to_timedelta(3600, unit="s"))

    # Next end date; if no end date provided, set to an hour after start_date
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    elif not isinstance(end_date, datetime.datetime):
        end_date = start_date + pd.to_timedelta(3600, unit="s")

    return start_date, end_date


def merge_csv_files(output_directory, path):
    """
    Merge all the CSV files inside path into a single file.

    :returns: path of the merged file.
    """
    LOGGER.info('Merging all the csv files into one CSV file.')
    merged_file = os.path.join(output_directory, 'merged_frames.csv')
    for filename in glob.glob(os.path.join(path, 'demod*/*.csv')):
        with open(filename, 'r') as source:
            content = source.read()
            with open(merged_file + '.tmp', 'a') as target:
                target.write(content)

    if os.path.exists(merged_file + '.tmp'):
        pass
    else:
        LOGGER.warning(' '.join([
            'There are no CSV files to merge.  This can happen',
            'if the time range specified had no frames to download,'
            'or if the downloaded files were deleted or modified',
            'during exeution.'
        ]))
        raise NoCSVFilesToMerge

    with open(merged_file + '.tmp', 'r') as source:
        with open(merged_file, 'w') as target:
            for line in sorted(source):
                target.write(line)
    os.remove(merged_file + '.tmp')
    LOGGER.info('Merge Completed')

    return merged_file


def data_fetch(norad_id, output_directory, start_date, end_date):
    """
    Fetch data of the sat with the given Norad ID gathered between start_date
    and end_date. Data is retrieved from SatNOGS database using Glouton.

    :returns: path of the file that contains the fetched data.
    """

    # Creating a new subdirectory to output directory
    # to collect glouton's data. Using start date to name it.
    cwd_path = os.path.join(output_directory,
                            "data_" + start_date.strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(cwd_path):
        os.mkdir(cwd_path)

    # Preparing glouton command configuration
    # pylint: disable-msg=unexpected-keyword-arg
    glouton_conf = ProgramCmd(norad_id=norad_id,
                              ground_station_id=None,
                              start_date=start_date,
                              end_date=end_date,
                              observation_status=None,
                              working_dir=cwd_path,
                              archives=False,
                              waterfalls=False,
                              demoddata=True,
                              archive_modules=None,
                              demoddata_modules=["CSV"],
                              waterfall_modules=None,
                              user=None,
                              transmitter_uuid=None,
                              transmitter_mode=None,
                              transmitter_type=None,
                              frame_modules=None,
                              observer=None,
                              app_source=None,
                              transmitter=None,
                              archive_end_modules=None,
                              demoddata_end_modules=None,
                              waterfall_end_modules=None,
                              frame_end_modules=None)

    # Running glouton data collection
    try:
        obs = ObservationsService(glouton_conf)
        obs.extract()
    except Exception as eee:  # pylint: disable=W0703
        LOGGER.error("data collection: %s", eee)

    LOGGER.info('Saving the dataframes in directory: %s', output_directory)
    try:
        return merge_csv_files(output_directory, cwd_path)
    except NoCSVFilesToMerge:
        return ""


def build_decoded_file_path(directory):
    """Return path to decoded files within directory

    :param directory: full path to directory for decoded frames

    :returns: path of the file that contains the decoded data.
    """
    return os.path.join(directory, 'decoded_frames.json')


def data_merge_and_decode(decoder,
                          output_directory,
                          new_frames_file="",
                          ignore_errors=False):
    """
    Decode the data found in frames_file using the given decoder. Put it in
    output_directory.

    :param decoder: decoder to use
    :param output_directory: where to put output
    :param new_frames_file: file to put new frames in
    :param ignore_errors: ignore errors when decoding frames

    :returns: path of the file that contains the decoded data.
    """

    # Using satnogs-decoders to decode the CSV files containing
    # multiple dataframes and store them as JSON objects.

    decoded_file = build_decoded_file_path(output_directory)

    if new_frames_file == "":
        LOGGER.info('No new frames to decode and merge')
    else:
        LOGGER.info('Starting decoding and merging of the new frames')
        decode_cmd = build_decode_cmd(new_frames_file, decoded_file, decoder)
        LOGGER.debug('Decode command: %s', decode_cmd)
        try:
            proc3 = subprocess.Popen(decode_cmd,
                                     shell=True,
                                     cwd=output_directory)
            proc3.wait()
            if proc3.returncode != 0 and ignore_errors is False:
                LOGGER.error(' '.join([
                    'decode_multiple cmd error.',
                    'You can choose to ignore it by passing',
                    '--ignore_errors flag in cmd',
                ]))
                raise DecodeMultipleFailed
            LOGGER.info('Decoding of data finished.')
        except subprocess.CalledProcessError as err:
            LOGGER.error('Error running %s: %s', decode_cmd, err)

    if os.path.exists(decoded_file):
        LOGGER.info('Decoded data stored at %s', decoded_file)
        return decoded_file

    LOGGER.error(' '.join([
        'There is no file of decoded frames at ' + decoded_file + '.',
        'This can happen if the time range specified had no frames',
        'to download, and you have not imported frames already.',
        'You may want to specify a different time range',
        'with the --start_date and --end_date options, or import',
        'frames downloaded directly from SatNOGS. This',
        'can also arise if the downloaded files have been',
        'deleted or modified during execution.'
    ]))
    raise NoDecodedFramesFile


def fetch_or_import(import_file, satellite, start_date, end_date, cache_dir):
    """
    Its a import_file validation function,
    it checks if variable import_file variable is empty or not,
    if its not empty then it checks if the file exists or not,
    if the file does not exit, it will raise an error.

    :param import_file: file to be checked if exists or not
    :param satellite: satellite whose data is to be fetch
    :param start_date: start date of data to fetch
    :param end_date: end date of data to fetch
    :param cache_dir: where temp output data should go
    returns: file if exists
    """
    # Retrieve, decode and normalize frames
    if import_file is None:
        # Converting dates into datetime objects
        start_date, end_date = build_start_and_end_dates(start_date, end_date)
        LOGGER.info('Fetch period: %s to %s', start_date, end_date)

        new_frames_file = data_fetch(satellite.norad_id, cache_dir, start_date,
                                     end_date)
    else:
        # If file is specified, retrieve from file
        if not os.path.isfile(import_file):
            raise SpecifiedImportFileDoesNotExist(
                'Import file does not exist.')
        new_frames_file = import_file
    return new_frames_file


def data_normalize(normalizer, frame_list):
    """
    Normalize the data found in frame_list using the given normalizer.

    :returns: list of normalized frames
    """
    # Normalize values
    normalized_frames = []
    frame_count = 0
    for frame in frame_list:
        frame_count += 1
        frame_norm = normalizer.normalize(frame)
        if not normalizer.validate_frame(frame_norm):
            LOGGER.debug("Skipping frame %d because validation failed",
                         frame_count)
            continue
        normalized_frames.append(frame_norm)
    return normalized_frames


def files_in_current_dir():
    """
    returns: list of csv and json files in the current directory.
    """
    candidate_files = [
        f for f in os.listdir()
        if os.path.isfile(f) and (f[-3:] == 'csv' or f[-4:] == 'json')
    ]
    return candidate_files


def load_frames_from_json_file(file):
    """Load frames from a JSON file.

    :param file: a JSON file
    :returns: a list of frames
    """
    with open(file) as f_handle:
        try:
            # pylint: disable=W0108
            decoded_frame_list = json.load(f_handle)
        except json.JSONDecodeError:
            LOGGER.error("Cannot load %s - is it a valid JSON document?", file)
            raise json.JSONDecodeError

    return decoded_frame_list


# pylint: disable-msg=too-many-arguments
def fetch_normalized_telemetry(satellite,
                               start_date,
                               end_date,
                               cache_dir,
                               import_file,
                               skip_normalizer=False,
                               ignore_errors=False):
    """
    Fetch, decode and normalize the telemetry

    :param satellite: Named tuple containing info about the satellite
    :type satellite: collections.namedtuple
    :param start_date: Start date of data to fetch
    :type start_date: str
    :param end_date: End date of data to fetch
    :type end_date: str
    :param cache_dir: Where temporary output data should go
    :type cache_dir: str, os.path
    :param import_file: File containing data frames to import
    :type import_file: str, os.path
    :return: Normalized space weather frames
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Try fetching data (from glouton/file)
    try:
        new_frames_file = fetch_or_import(import_file, satellite, start_date,
                                          end_date, cache_dir)
    except SpecifiedImportFileDoesNotExist:
        LOGGER.critical(' '.join([
            'Import file does not exist.', 'Some Suggestions:- ',
            ' '.join(files_in_current_dir())
        ]))
        sys.exit(1)

    # decode the frames fetched from glouton
    decoded_frames_file = data_merge_and_decode(satellite.decoder, cache_dir,
                                                new_frames_file, ignore_errors)
    decoded_frame_list = load_frames_from_json_file(decoded_frames_file)

    try:
        if skip_normalizer:
            # skip the normalization to SI units
            LOGGER.info("Skip normalizer passed")
            normalizer = load_normalizer("Dummy")()

            # Find the list of all fields in telemetry
            list_of_keys = set().union(*(frame["fields"].keys()
                                         for frame in decoded_frame_list))

            # Create the dummy normalizer
            normalizer.create_dummy_normalizer(list_of_keys)
            LOGGER.info('Loaded dummy normalizer')
        else:
            # Normalize the frames to SI units
            # Creating a normalizer object here
            normalizer = load_normalizer(satellite)()
            LOGGER.info('Loaded normalizer=%s', satellite.normalizer)
    except Exception as exception:
        LOGGER.error("Can't load satellite normalizer: %s", exception)
        raise exception

    normalized_frames = data_normalize(normalizer, decoded_frame_list)

    return normalized_frames
