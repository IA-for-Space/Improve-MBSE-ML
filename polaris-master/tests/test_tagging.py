"""pytest framework for ColumnTagging
"""

from polaris.data.fetched_data_preprocessor import FetchedDataPreProcessor


def test_column_tagging_tag_columns(polaris_m_frames_dataset_obj):
    """Test that tagging return a columns_tag dictionary
    """
    dataset = polaris_m_frames_dataset_obj
    tagger = FetchedDataPreProcessor()
    tagger.tag_columns(dataset)
    analyze = tagger.analysis
    assert analyze is not None
    assert analyze == {
        'column_tags': {
            'dest_callsign': 'constant',
            'src_callsign': 'constant',
            'src_ssid': 'constant',
            'dest_ssid': 'constant',
            'ctl': 'constant',
            'pid': 'constant',
            'type': 'constant',
            'bat1_volt': 'variable',
            'bat1_temp': 'variable',
            'bat1_flags': 'variable',
            'bat1_ctlflags': 'constant',
            'time': 'variable'
        },
        'feature_columns': None
    }
