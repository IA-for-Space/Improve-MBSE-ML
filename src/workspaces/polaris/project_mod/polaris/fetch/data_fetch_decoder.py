"""
Module for fetching and decoding telemetry data
"""
import json
import logging
import os
import re
import sys
from collections import namedtuple

from polaris.data.fetched_data_preprocessor import FetchedDataPreProcessor
from polaris.dataset.dataset import PolarisDataset
from polaris.fetch.fetch_import_sw import fetch_preprocessed_sw
from polaris.fetch.fetch_import_telemetry import fetch_normalized_telemetry, \
    load_frames_from_json_file

Satellite = namedtuple('Satellite',
                       ['norad_id', 'name', 'decoder', 'normalizer'])

SATELLITE_DATA_FILE = 'satellites.json'
SATELLITE_DATA_DIR = os.path.dirname(__file__)
_SATELLITES = json.loads(
    open(os.path.join(SATELLITE_DATA_DIR, SATELLITE_DATA_FILE)).read(),
    object_hook=lambda d: Satellite(d['norad_id'], d['name'], d['decoder'], d[
        'normalizer']))

LOGGER = logging.getLogger(__name__)


class NoSuchSatellite(Exception):
    """Raised when we can't identify the satellite requested """


class NoDecoderForSatellite(Exception):
    """Raised when we have no decoder """


class SatelliteNamesNotMatching(Exception):
    """Raised when satellite name of fetched telemetry database
    does not match the existing satellite name in the output file
    """


def get_times_from_frames_list(list_of_frames, key='time'):
    """Gets the time list from decoded_frames list

    :param list_of_frames: List containing the decoded frames
    :type list_of_frames: list
    :param key: Name of the time key in the dictionares, defaults to 'time'
    :type key: str, optional
    :return: List of timestamps taken from list_of_frames
    :rtype: list
    """
    return [frame[key] for frame in list_of_frames]


def write_or_merge(dataset, file, strategy):
    """Write dataset to output_file; if output file already exists, follow
    strategy: overwrite, merge or error.

    """
    def write_dataset(dataset, file):
        with open(file, 'w') as f_handle:
            f_handle.write(dataset.to_json())

    file_exists = os.path.exists(file)

    if strategy == 'overwrite':
        LOGGER.info('Overwriting existing file')
        write_dataset(dataset, file)
    elif strategy == 'error' and file_exists is True:
        raise FileExistsError(
            'Output file already exists, refusing to overwrite.')
    else:
        # Default strategy is merge.

        # Take copy of dataset, so that we don't change the original
        # object.
        dataset_for_writing = PolarisDataset(metadata=dataset.metadata,
                                             frames=dataset.frames)
        if file_exists is True:
            try:
                LOGGER.debug('Trying to load dataset from %s', file)
                existing_dataset = load_frames_from_json_file(file)
                # Verify that fetch encoder matches
                # existing encoder in the output file.
                if (existing_dataset['metadata']['satellite_name'] !=
                        dataset.metadata['satellite_name']):
                    raise SatelliteNamesNotMatching(' '.join([
                        'Satellite name used does not match satellite_name',
                        'in the existing output file, refusing to merge'
                    ]))
                dataset_for_writing.frames = existing_dataset[
                    'frames'] + dataset.frames
            except json.JSONDecodeError:
                LOGGER.info("File exists but cannot parse it")
        write_dataset(dataset_for_writing, file)


def combine_frames(satellite_frames, sw_frames):
    """
    Combines normalized satellite_frames with normalized sw_frames

    :param satellite_frames: List of normalized frames
    :type satellite_frames: list
    :param sw_frames: Dictionary with key as the space_weather index, value as
        list of space_weather frames for that index
    :type sw_frames: dict
    :return: List of frames where the fields are from both sw_frames as well as
        satellite_frames
    :rtype: list
    """
    LOGGER.info("Combining space weather and telemetry frames")
    combined_frames = []
    # satellite_frames is a list
    for frame_no, frame in enumerate(satellite_frames):
        # sw_frames is a dict of lists
        for sw_frame in sw_frames.values():
            frame['fields'] = {
                **frame['fields'],
                **sw_frame[frame_no]['fields']
            }

        combined_frames.append(frame)

    return combined_frames


def normalize_satname(sat_name):
    """
    Normalize satellite name for comparison or searching

    :param sat_name: Satellite name
    """
    return re.sub('[^A-Za-z0-9]+', '', sat_name).lower()


def find_alternatives(sat_name, list_of_satellites):
    """
    Find an alternative for sat_name from dict_of_satellites

    :param sat_name: Name of satellite to check
    :param list_of_satellites: Dictionary containing
           all satellite details
    """
    for sat_present in list_of_satellites:
        if normalize_satname(sat_name) == normalize_satname(sat_present.name):
            return sat_present.name

    return None


def find_satellite(sat, sat_list):
    """Find a match for a given satellite in a list of satellites """
    for candidate in sat_list:
        if sat in (candidate.name, candidate.norad_id):
            LOGGER.info('Satellite: id=%s name=%s decoder=%s',
                        candidate.norad_id, candidate.name, candidate.decoder)
            LOGGER.info('selected decoder=%s', candidate.decoder)
            if candidate.decoder is None:
                LOGGER.error('Satellite %s not supported!', sat)
                raise NoDecoderForSatellite
            return candidate
    raise NoSuchSatellite


# pylint: disable-msg=too-many-arguments
# pylint: disable-msg=too-many-locals
def data_fetch_decode_normalize(sat,
                                start_date,
                                end_date,
                                output_file,
                                cache_dir,
                                import_file,
                                existing_output_file_strategy,
                                skip_normalizer=False,
                                ignore_errors=False,
                                **kwargs):
    """
    Main function to download and decode satellite telemetry.

    :param sat: a NORAD ID or a satellite name.
    :param start_date: start date of data to fetch
    :param end_date: end date of data to fetch
    :param output_file: where output should go
    :param cache_dir: where temp output data should go
    :param import_file: file containing data frames to import
    :param existing_output_file_strategy: what to do with existing
           output files: merge, overwrite or error.
    :param skip_normalizer: skip normalizing of data
    :param ignore_errors: ignore errors when decoding frames
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Check if satellite info available
    try:
        satellite = find_satellite(sat, _SATELLITES)
    except Exception as exception:
        LOGGER.error("Can't find satellite or decoder: %s", exception)
        LOGGER.info("You can check for your satellite 'name' by using command "
                    "`polaris fetch --list_supported_satellites` ")
        # Check if there is a satellite with a similar name
        alt_sat = find_alternatives(sat, _SATELLITES)
        if alt_sat is not None:
            LOGGER.info("Did you mean: %s?", alt_sat)
        raise exception

    # Fetch normalized telemetry
    normalized_telemetry = fetch_normalized_telemetry(satellite, start_date,
                                                      end_date, cache_dir,
                                                      import_file,
                                                      skip_normalizer,
                                                      ignore_errors)

    # Get timestamps for which space weather needs to be extracted
    time_list = get_times_from_frames_list(normalized_telemetry)

    # Get preprocessed space weather
    sw_columns_names, preprocessed_sw = fetch_preprocessed_sw(
        start_date, end_date, cache_dir, time_list, sat, **kwargs)

    # Combine the preprocessed space weather and normalized telemetry
    combined_frames = combine_frames(normalized_telemetry, preprocessed_sw)
    polaris_dataset = PolarisDataset(
        metadata={
            "satellite_norad": satellite.norad_id,
            "satellite_name": satellite.name,
            "total_frames": len(normalized_telemetry),
        },
        frames=combined_frames,
    )

    # Tag columns as variable, status and constant
    LOGGER.info('Tagging columns')
    tagger = FetchedDataPreProcessor()
    tagger.tag_columns(polaris_dataset)
    tagger.add_columns_in_feature_list(sw_columns_names)
    polaris_dataset.metadata["analysis"] = tagger.analysis
    LOGGER.info('Tagging Completed')

    # Write all the data
    try:
        write_or_merge(polaris_dataset, output_file,
                       existing_output_file_strategy)
        LOGGER.info('Output file %s', output_file)
    except FileExistsError:
        LOGGER.critical(' '.join([
            'Output file exists and told not to overwrite it.',
            'Remove it, or try a different argument',
            'for --existing-output-file-strategy.'
        ]))
        sys.exit(1)
    except SatelliteNamesNotMatching:
        LOGGER.critical(' '.join([
            'The satellite name you entered does not match',
            'the satellite name in your existing output file.'
        ]))
        sys.exit(1)
