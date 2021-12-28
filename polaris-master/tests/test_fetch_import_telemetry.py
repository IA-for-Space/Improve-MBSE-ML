"""
pytest testing framework for fetch_import_telemetry
"""
import datetime
import os

import pandas as pd
import pytest

from contrib.normalizers.common import Field, Normalizer
from polaris.fetch import fetch_import_telemetry
from polaris.fetch.data_fetch_decoder import _SATELLITES, find_satellite

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "test_data")

SINGLE_FRAME = [
    {
        "time": "2019-01-01 00:00:00",
        "fields": {
            'example_telemetry': 0
        }
    },
]

MULTIPLE_FRAMES = [{
    "time": "2019-01-01 00:00:00",
    "fields": {
        'example_telemetry': 0
    }
}, {
    "time": "2019-01-01 01:00:00",
    "fields": {
        'example_telemetry': 1
    }
}, {
    "time": "2019-01-01 02:00:00",
    "fields": {
        'example_telemetry': 2
    }
}]

SINGLE_AX25_FRAME = [
    {
        "time": "2019-01-01 00:00:00",
        "fields": {
            'example_telemetry': 0,
            'src_callsign': {
                'value': 'A1B2C3',
                'unit': None,
            }
        }
    },
]

MULTIPLE_AX25_FRAMES = [{
    "time": "2019-01-01 00:00:00",
    "fields": {
        'example_telemetry': 0,
        'src_callsign': {
            'value': 'A1B2C3',
            'unit': None,
        }
    }
}, {
    "time": "2019-01-01 00:01:00",
    "fields": {
        'example_telemetry': 1,
        'src_callsign': {
            'value': 'Z9Y8X7',
            'unit': None,
        }
    }
}, {
    "time": "2019-01-01 00:02:00",
    "fields": {
        'example_telemetry': 0,
        'src_callsign': {
            'value': 'A1B2C3',
            'unit': None,
        }
    }
}]


class FixtureNormalizer(Normalizer):
    """Normalizer fixture for pytest
    """
    def __init__(self):
        super().__init__()
        self.normalizers = [
            Field('example_telemetry', lambda x: x, None, 'Example Telemetry')
        ]


class FixtureNormalizerWithValidator(FixtureNormalizer):
    """Normalizer fixture with a validator
    """
    def validate_frame(self, frame):
        try:
            return frame['fields']['src_callsign']['value'].lower() == 'a1b2c3'
        except (KeyError, AttributeError):
            return False


def test_load_normalizer_no_normalizer(satellite_list):
    """Test no_normlizer path for find_satellite()
    """
    test_satellite = satellite_list[2]
    with pytest.raises(fetch_import_telemetry.NoNormalizerForSatellite):
        _ = fetch_import_telemetry.load_normalizer(test_satellite)


def test_build_dates_from_string():
    """Test dates conversion for build_start_and_end_dates()
    """
    start_date_str = '2019-08-14'
    end_date_str = '2019-08-16'
    start_date, end_date = fetch_import_telemetry.build_start_and_end_dates(
        start_date_str, end_date_str)
    assert end_date - start_date == pd.to_timedelta(2, unit="D")

    start_date_str = '2019-08-14 11:00:00'
    end_date_str = '2019-08-16 11:00:00'
    start_date, end_date = fetch_import_telemetry.build_start_and_end_dates(
        start_date_str, end_date_str)
    assert end_date - start_date == pd.to_timedelta(2, unit="D")


def test_build_dates_from_default():
    """Test default dates generation for build_start_and_end_dates()
    """
    start_date, end_date = fetch_import_telemetry.build_start_and_end_dates(
        None, None)
    assert end_date - start_date == pd.to_timedelta(3600, unit="s")


def test_build_dates_from_mix():
    """Test default dates generation for build_start_and_end_dates()
       in case called from code with mixture of input types.
    """
    start_date_x = '2019-11-14 11:00:00'
    end_date_x = pd.to_datetime('2019-11-16 11:00:00')
    start_date, end_date = fetch_import_telemetry.build_start_and_end_dates(
        start_date_x, end_date_x)
    assert end_date - start_date == pd.to_timedelta(2, unit="D")

    start_date_x = datetime.datetime(2019, 11, 14, 11)
    end_date_x = pd.to_datetime('2019-11-16 11:00:00')
    start_date, end_date = fetch_import_telemetry.build_start_and_end_dates(
        start_date_x, end_date_x)
    assert end_date - start_date == pd.to_timedelta(2, unit="D")


def test_data_normalize_empty_list():
    """Test data_normalize() with empty list of frames
    """
    frame_list = []

    normalized_frames = fetch_import_telemetry.data_normalize(
        FixtureNormalizer(), frame_list)
    assert normalized_frames == []


def test_data_normalize_happy_path_single_frame():
    """Test data_normalize() happy path with single frame
    """

    normalized_frames = fetch_import_telemetry.data_normalize(
        FixtureNormalizer(), SINGLE_FRAME)

    assert len(normalized_frames) == 1
    assert normalized_frames[0]['fields'] == {
        'example_telemetry': {
            'value': 0,
            'unit': None
        }
    }


def test_data_normalize_happy_path_multiple_frames():
    """Test data_normalize() happy path with multiple_frames
    """

    normalized_frames = fetch_import_telemetry.data_normalize(
        FixtureNormalizer(), MULTIPLE_FRAMES)

    assert len(normalized_frames) == 3
    for i in range(len(normalized_frames)):  # pylint: disable=C0103,C0200
        assert normalized_frames[i]['fields'] == {
            'example_telemetry': {
                'value': i,
                'unit': None
            }
        }


def test_data_normalize_validator_happy_path():
    """Test data_normalize validator happy path
    """

    normalized_frames = fetch_import_telemetry.data_normalize(
        FixtureNormalizerWithValidator(), SINGLE_AX25_FRAME)

    assert len(normalized_frames) == 1


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "merged_frames.csv"))
def test_data_normalize_skip_normalizer(datafiles):
    """Test data_normalize while skipping the normalizer
    """
    sat = "LightSail-2"
    satellite = find_satellite(sat, _SATELLITES)
    import_file = str(datafiles / "merged_frames.csv")
    normalized_frames = fetch_import_telemetry.fetch_normalized_telemetry(
        satellite, None, None, datafiles, import_file, True)

    assert len(normalized_frames) != 0


def test_data_normalize_validator_happy_path_multiple_frames():
    """Test data_normalize validator happy path with multiple_frames
    """

    normalized_frames = fetch_import_telemetry.data_normalize(
        FixtureNormalizerWithValidator(), MULTIPLE_AX25_FRAMES)

    assert len(normalized_frames) == 2
