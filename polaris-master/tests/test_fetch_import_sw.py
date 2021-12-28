"""
Module to test fetch_import_sw
"""
import datetime
import os

import pytest
from vinvelivaanilai.space_weather import sw_extractor

from polaris.fetch import fetch_import_sw

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "test_fetch_import_sw")


def test_load_sw_normalizer_happy():
    """Test happy case for load_sw_normalizer()
    """
    for index in fetch_import_sw.SUPPORTED_INDICES:
        _ = fetch_import_sw.load_sw_normalizer(index)


def test_load_sw_normalizer_sad():
    """Test sad case for load_sw_normalizer()
    """
    with pytest.raises(Exception):
        _ = fetch_import_sw.load_sw_normalizer("what_is_this")


@pytest.mark.datafiles()
def test_fetch_import_sw_happy(datafiles):
    """Test happy case for fetching space weather
    """
    # Fetching for short period only since vinvelivaanilai already tests
    start_date = datetime.datetime(year=2019, month=3, day=10)
    end_date = datetime.datetime(year=2019, month=3, day=30)

    cache_dir = str(datafiles)
    fetched_data = fetch_import_sw.fetch_sw(start_date, end_date, cache_dir)

    for data in fetched_data.values():
        # We want to make sure there are no nan/null values
        # Null values cause problems for the learn module
        assert not data.isnull().any().any()


@pytest.mark.datafiles()
def test_fetch_import_sw_sad(datafiles):
    """Test sad case for fetching space weather
    """
    start_date = datetime.datetime(year=2018, month=3, day=28)
    end_date = datetime.datetime(year=2019, month=5, day=17)

    cache_dir = str(datafiles)

    with pytest.raises(ValueError):
        # Do not allow unsupported values inside
        _ = fetch_import_sw.fetch_sw(start_date, end_date, cache_dir, "bad")


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "2019_DGD.txt"),
                       os.path.join(FIXTURE_DIR, "2019_DPD.txt"),
                       os.path.join(FIXTURE_DIR, "2019_DSD.txt"))
def test_fetch_nearest_sw(datafiles, time_list):
    """Test fetch_nearest_sw
    """
    dgd_data = sw_extractor.extract_data_regex("DGD",
                                               str(datafiles / "2019_DGD.txt"))
    dpd_data = sw_extractor.extract_data_regex("DPD",
                                               str(datafiles / "2019_DPD.txt"))
    dsd_data = sw_extractor.extract_data_regex("DSD",
                                               str(datafiles / "2019_DSD.txt"))

    sw_data = {
        "DGD": dgd_data,
        "DPD": dpd_data,
        "DSD": dsd_data,
    }

    _ = fetch_import_sw.fetch_nearest_sw(sw_data, time_list)


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "2019_DGD.txt"),
                       os.path.join(FIXTURE_DIR, "2019_DPD.txt"),
                       os.path.join(FIXTURE_DIR, "2019_DSD.txt"))
def test_dataframe_to_decoded(datafiles):
    """Test dataframe_to_decoded
    """
    dgd_data = sw_extractor.extract_data_regex("DGD",
                                               str(datafiles / "2019_DGD.txt"))
    dpd_data = sw_extractor.extract_data_regex("DPD",
                                               str(datafiles / "2019_DPD.txt"))
    dsd_data = sw_extractor.extract_data_regex("DSD",
                                               str(datafiles / "2019_DSD.txt"))

    sw_data = {
        "DGD": dgd_data,
        "DPD": dpd_data,
        "DSD": dsd_data,
    }

    for data in sw_data.values():
        _ = fetch_import_sw.dataframe_to_decoded(data)
