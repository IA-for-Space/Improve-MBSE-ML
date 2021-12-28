"""
pytest testing framework for fetch module
"""
import os
from contextlib import contextmanager
from copy import deepcopy

import pytest

from polaris.dataset.dataset import PolarisDataset
from polaris.fetch import data_fetch_decoder

FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))


def test_find_satellite_happy(satellite_list):
    """Test happy path for find_satellite()
    """
    # test_satellite = 'LightSail-2'
    test_satellite = 'ExampleSat'
    sat = data_fetch_decoder.find_satellite(test_satellite, satellite_list)
    assert isinstance(sat, data_fetch_decoder.Satellite)


def test_find_satellite_sad(satellite_list):
    """Test sad path for find_satellite()
    """
    test_satellite = 'DoesNotExist'
    with pytest.raises(data_fetch_decoder.NoSuchSatellite):
        _ = data_fetch_decoder.find_satellite(test_satellite, satellite_list)


def test_find_satellite_no_decoder(satellite_list):
    """Test no_decoder path for find_satellite()
    """
    test_satellite = 'NoDecoderSatellite'
    with pytest.raises(data_fetch_decoder.NoDecoderForSatellite):
        _ = data_fetch_decoder.find_satellite(test_satellite, satellite_list)


def test_find_alternatives_happy(satellite_list):
    """Test happy path for find_alternatives
    """
    test_satellite = 'EXAMPLEsat'
    alt_sat = data_fetch_decoder.find_alternatives(test_satellite,
                                                   satellite_list)
    act_sat = 'ExampleSat'
    assert alt_sat == act_sat


def test_find_alternatives_sad(satellite_list):
    """Test sad path for find_alternatives
    """
    test_satellite = 'DoesNotExist'
    alt_sat = data_fetch_decoder.find_alternatives(test_satellite,
                                                   satellite_list)
    assert alt_sat is None


@contextmanager
def does_not_raise():
    """Utility function for tests that does not yield an exception.
    """
    yield


def create_fixture_file(data, filename):
    """Utility function for tests that writes out a Polaris dataset.
    """
    with open(filename, 'w') as f_handle:
        f_handle.write(data.to_json())


@pytest.mark.parametrize(
    "file, strategy, expectation",
    [('exists.json', 'merge', does_not_raise()),
     ('exists.json', 'overwrite', does_not_raise()),
     ('exists.json', 'error', pytest.raises(FileExistsError))])
def test_write_or_merge_existing_files(file, strategy, expectation, tmp_path,
                                       polaris_dataset_obj):
    """Test write_or_merge file writes with existing files
    """
    with expectation:
        fullpath = tmp_path / file
        create_fixture_file(polaris_dataset_obj, fullpath)
        data_fetch_decoder.write_or_merge(polaris_dataset_obj,
                                          fullpath.as_posix(), strategy)


@pytest.mark.parametrize(
    "file, strategy, expectation",
    [('does_not_exist.json', 'merge', does_not_raise()),
     ('does_not_exist.json', 'overwrite', does_not_raise()),
     ('does_not_exist.json', 'error', does_not_raise())])
def test_write_or_merge_non_existing_files(file, strategy, expectation,
                                           tmp_path, polaris_dataset_obj):
    """Test write_or_merge file writes with non-existing files
    """
    with expectation:
        fullpath = tmp_path / file
        data_fetch_decoder.write_or_merge(polaris_dataset_obj,
                                          fullpath.as_posix(), strategy)


@pytest.mark.parametrize("strategy, exception_expected, new_frames_multiple",
                         [('merge', does_not_raise(), 2),
                          ('overwrite', does_not_raise(), 1),
                          ('error', pytest.raises(FileExistsError), 1)])
def test_write_or_merge_frame_length(strategy, exception_expected,
                                     new_frames_multiple, tmp_path,
                                     polaris_dataset_obj):
    """Test write_or_merge data writes with existing files
    """
    fullpath = tmp_path / 'frame_length_test.json'
    with exception_expected:
        create_fixture_file(polaris_dataset_obj, fullpath)
        data_fetch_decoder.write_or_merge(polaris_dataset_obj,
                                          fullpath.as_posix(), strategy)
    old_frame_length = len(polaris_dataset_obj.frames)
    expected_new_frame_length = old_frame_length * new_frames_multiple
    new_obj = PolarisDataset()
    with open(fullpath.as_posix()) as f_handle:
        new_obj.from_json(f_handle.read())
    assert len(new_obj.frames) == expected_new_frame_length


@pytest.mark.parametrize(
    "strategy, exception_expected",
    [('merge', pytest.raises(data_fetch_decoder.SatelliteNamesNotMatching)),
     ('overwrite', does_not_raise()),
     ('error', pytest.raises(FileExistsError))])
def test_write_or_merge_decoder_name(strategy, exception_expected, tmp_path,
                                     polaris_dataset_obj):
    """Test write_or_merge data writes with different decoder name
    """
    fullpath = tmp_path / 'frames.json'
    create_fixture_file(polaris_dataset_obj, fullpath)

    decoder_name = 'My Satellite'
    new_obj = deepcopy(polaris_dataset_obj)
    new_obj.metadata['satellite_name'] = decoder_name
    with exception_expected:
        data_fetch_decoder.write_or_merge(new_obj, fullpath.as_posix(),
                                          strategy)


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "test_data"))
def test_data_fetch_decode_normalize(datafiles):
    """Test data_fetch_decode_normalize. The data is already present in the
    folder so there should not be any load on the servers!
    """
    sat = "LightSail-2"
    start_date = "2020-03-10"
    end_date = "2020-03-11"
    output_file = str(datafiles / "test_normalized_frames.json")
    cache_dir = str(datafiles)
    import_file = None
    existing_output_file_strategy = "overwrite"

    data_fetch_decoder.data_fetch_decode_normalize(
        sat, start_date, end_date, output_file, cache_dir, import_file,
        existing_output_file_strategy)
