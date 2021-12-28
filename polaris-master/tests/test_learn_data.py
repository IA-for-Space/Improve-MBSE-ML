"""
Module for testing data.readers.py script.
"""

import json

import pytest

import polaris.data.readers as pldr


def test_read_polaris_data_missing_file():
    """Test reading polaris data, missing file
    """
    with pytest.raises(FileNotFoundError):
        _, _ = pldr.read_polaris_data("/tmp/tmp/tmp/a/b/a/b/NOTINSPACE.csv")
    with pytest.raises(FileNotFoundError):
        _, _ = pldr.read_polaris_data("/tmp/tmp/tmp/a/b/a/b/NOTINSPACE.json")


def test_read_polaris_data_unknown_format():
    """Test reading polaris data, unknown format
    """
    with pytest.raises(pldr.PolarisUnknownFileFormatError):
        _, _ = pldr.read_polaris_data("/tmp/tmp/tmp/a/b/a/b/NOTINSPACE")


def test_read_polaris_data_from_json_happy_path(polaris_dataset_json,
                                                tmp_path):
    """Test reading polaris data from json, happy path
    """
    fullpath = tmp_path / "dataset.json"
    with open(fullpath, 'w') as f_handle:
        f_handle.write(polaris_dataset_json)
    metadata, input_data = pldr.read_polaris_data_from_json(fullpath)
    assert metadata['satellite_name'] == "LightSail-2"

    dist_dataset = json.loads(polaris_dataset_json)

    # pylint: disable=invalid-name
    for i in range(input_data.shape[0]):
        input_data_row = input_data.iloc[i]
        dist_dataset_row = dist_dataset['frames'][i]
        for col in input_data.columns:
            if col == 'time':
                # Used for index, won't be in the fields
                continue
            assert input_data_row[col] == dist_dataset_row['fields'][col][
                'value']
