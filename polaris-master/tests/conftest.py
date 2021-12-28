"""
Fixtures for pytest tests
"""

import datetime
import json
import os

import pandas as pd
import pytest

from polaris.convert.gexf import GEXFConverter
from polaris.dataset.dataset import PolarisDataset
from polaris.dataset.frame import PolarisFrame
from polaris.dataset.metadata import PolarisMetadata
from polaris.fetch import data_fetch_decoder


@pytest.fixture
def satellite_list():
    """List of satellites to be used in testing
    """
    return [
        data_fetch_decoder.Satellite(name='ExampleSat',
                                     norad_id='12345',
                                     decoder='ExampleDecoder',
                                     normalizer='ExampleNormalizer'),
        data_fetch_decoder.Satellite(name='NoDecoderSatellite',
                                     norad_id='67890',
                                     decoder=None,
                                     normalizer='ExampleNormalizer'),
        data_fetch_decoder.Satellite(name='NoNormalizerSatellite',
                                     norad_id='67890',
                                     decoder='ExampleNormalizer',
                                     normalizer=None)
    ]


POLARIS_METADATA_DICT = {
    'date': 1567460994,
    'cli_options': '',
    'satellite_norad': '44420',
    'satellite_name': 'LightSail-2',
    'total_frames': 2
}


@pytest.fixture
def polaris_metadata_dict():
    """PolarisMetadata dict to be used in testing
    """
    return POLARIS_METADATA_DICT


@pytest.fixture
def polaris_metadata_obj():
    """PolarisMetadata obj to be used in testing
    """
    polaris_metadata = PolarisMetadata(POLARIS_METADATA_DICT)
    return polaris_metadata


POLARIS_FRAME_DICT = {
    "time": "2019-07-21 20:43:57",
    "measurement": "",
    "tags": {
        "satellite": "",
        "decoder": "Lightsail2",
        "station": "",
        "observer": "",
        "source": "",
        "version": "0.15.1"
    },
    "fields": {
        "dest_callsign": {
            "value": "N6CP  ",
            "unit": None
        },
        "src_callsign": {
            "value": "KK6HIT",
            "unit": None
        },
        "src_ssid": {
            "value": 2,
            "unit": None
        },
        "dest_ssid": {
            "value": 1,
            "unit": None
        },
        "ctl": {
            "value": 3,
            "unit": None
        },
        "pid": {
            "value": 204,
            "unit": None
        },
        "type": {
            "value": 1,
            "unit": None
        },
        "bat1_volt": {
            "value": 3.875,
            "unit": "V"
        },
        "bat1_temp": {
            "value": 13.0,
            "unit": "degC"
        },
        "bat1_flags": {
            "value": 165,
            "unit": None
        },
        "bat1_ctlflags": {
            "value": 0,
            "unit": None
        }
    }
}

POLARIS_FRAME_DICT_FOR_TAGGING = {
    "time": "2019-07-21 15:42:50",
    "measurement": "",
    "tags": {
        "satellite": "",
        "decoder": "Lightsail2",
        "station": "",
        "observer": "",
        "source": "",
        "version": "0.15.1"
    },
    "fields": {
        "dest_callsign": {
            "value": "N6CP  ",
            "unit": None
        },
        "src_callsign": {
            "value": "KK6HIT",
            "unit": None
        },
        "src_ssid": {
            "value": 2,
            "unit": None
        },
        "dest_ssid": {
            "value": 1,
            "unit": None
        },
        "ctl": {
            "value": 3,
            "unit": None
        },
        "pid": {
            "value": 204,
            "unit": None
        },
        "type": {
            "value": 1,
            "unit": None
        },
        "bat1_volt": {
            "value": 2.932,
            "unit": "V"
        },
        "bat1_temp": {
            "value": 10.0,
            "unit": "degC"
        },
        "bat1_flags": {
            "value": 164,
            "unit": None
        },
        "bat1_ctlflags": {
            "value": 0,
            "unit": None
        }
    }
}


@pytest.fixture
def polaris_frame_dict():
    """PolarisFrame dict to be used in testing
    """
    return POLARIS_FRAME_DICT


POLARIS_FRAME_OBJ = PolarisFrame(POLARIS_FRAME_DICT)


@pytest.fixture
def polaris_frame_obj():
    """PolarisFrame object to be used in testing
    """
    return POLARIS_FRAME_OBJ


POLARIS_DATASET_OBJ = PolarisDataset(metadata=POLARIS_METADATA_DICT,
                                     frames=[POLARIS_FRAME_DICT])


@pytest.fixture
def polaris_dataset_obj():
    """PolarisDataset object to be used in testing
    """
    return POLARIS_DATASET_OBJ


POLARIS_FRAMES_DATASET_OBJ = PolarisDataset(
    metadata=POLARIS_METADATA_DICT,
    frames=[POLARIS_FRAME_DICT, POLARIS_FRAME_DICT_FOR_TAGGING])


@pytest.fixture
def polaris_m_frames_dataset_obj():
    """PolarisDataset object to be used in testing
    """
    return POLARIS_FRAMES_DATASET_OBJ


@pytest.fixture
def polaris_dataset_json():
    """PolarisDataset JSON to be used in testing
    """
    return """{
    "metadata": {
        "data_format_version": 1,
        "date": 1567460994,
        "cli_options": "",
        "satellite_norad": "44420",
        "satellite_name": "LightSail-2",
        "total_frames": 2
    },
    "frames": [
        {
            "time": "2019-07-21 20:43:57",
            "measurement": "",
            "tags": {
                "satellite": "",
                "decoder": "Lightsail2",
                "station": "",
                "observer": "",
                "source": "",
                "version": "0.15.1"
            },
            "fields": {
                "dest_callsign": {
                    "value": "N6CP  ",
                    "unit": null
                },
                "src_callsign": {
                    "value": "KK6HIT",
                    "unit": null
                },
                "src_ssid": {
                    "value": 2,
                    "unit": null
                },
                "dest_ssid": {
                    "value": 1,
                    "unit": null
                },
                "ctl": {
                    "value": 3,
                    "unit": null
                },
                "pid": {
                    "value": 204,
                    "unit": null
                },
                "type": {
                    "value": 1,
                    "unit": null
                },
                "bat1_volt": {
                    "value": 3.875,
                    "unit": "V"
                },
                "bat1_temp": {
                    "value": 13.0,
                    "unit": "degC"
                },
                "bat1_flags": {
                    "value": 165,
                    "unit": null
                },
                "bat1_ctlflags": {
                    "value": 0,
                    "unit": null
                }
            }
        }
    ]
}"""


@pytest.fixture
def polaris_config():
    """Polaris configuration JSON to be used in testing
    """
    return json.dumps({
        'file_layout': {
            'root_dir': '/tmp/polaris'
        },
        'satellite': {
            'name': 'LightSail-2',
            '_comment': 'This is a comment',
            'batch': {
                'fetch': True,
                'learn': True,
                'viz': False
            }
        }
    })


@pytest.fixture
def polaris_config_defaults():
    """Polaris configuration defaults dict to be used in testing
    """
    return {
        'file_layout': {
            'root_dir': '/default_root_dir'
        },
        'satellite': {
            'batch': {
                'fetch': True,
                'learn': True,
                'viz': False
            }
        }
    }


@pytest.fixture
def polaris_heatmap_fixture():
    """Polaris heatmap to be used in testing, plus testing info
    """
    feature_names = [
        'ZerothFeature', 'FirstFeature', 'SecondFeature', 'ThirdFeature'
    ]

    zeroth_col_data = [1.0, None, 0.01, 0.75]
    first_col_data = [0.75, 1.0, None, 0.01]
    second_col_data = [0.01, 0.75, 1.0, None]
    third_col_data = [None, 0.01, 0.75, 1.0]

    # The expected links are determined by PolarisGraph.to_heatmap()
    expected_links = [
        {
            'source': 'ZerothFeature',
            'target': 'FirstFeature',
            'value': 0.75
        },
        {
            'source': 'FirstFeature',
            'target': 'SecondFeature',
            'value': 0.75
        },
        {
            'source': 'SecondFeature',
            'target': 'ThirdFeature',
            'value': 0.75
        },
        {
            'source': 'ThirdFeature',
            'target': 'ZerothFeature',
            'value': 0.75
        },
    ]

    fixture = {}
    fixture['data'] = [
        zeroth_col_data, first_col_data, second_col_data, third_col_data
    ]
    fixture['columns'] = feature_names
    fixture['df'] = pd.DataFrame(fixture['data'], fixture['columns'],
                                 fixture['columns'])
    fixture['expected_links'] = expected_links

    return fixture


@pytest.fixture
def time_list():
    """Fixture for sample list of time
    """
    times = [
        datetime.datetime(year=2019, month=10, day=13, hour=10, minute=17),
        datetime.datetime(year=2019, month=10, day=24, hour=23),
        datetime.datetime(year=2019, month=11, day=3, hour=5, minute=17),
        datetime.datetime(year=2019, month=11, day=29, hour=1),
    ]
    return times


@pytest.fixture
def gexf_converter_min(tmp_path):
    """ GEXF converter with valid processed small graph input file
    """
    return GEXFConverter(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     'test_convert', 'graph_ok_min.json'),
        tmp_path / "test_save_min.gexf")
