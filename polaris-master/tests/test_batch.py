"""
Module for testing batch.py script.
"""

import subprocess
import time
from datetime import date, timedelta

import polaris.batch.batch as batch
from polaris.common.config import PolarisConfig

TODAY_TIMESTRUCT = time.gmtime()
TODAY_STRING = time.strftime('%Y-%m-%d', TODAY_TIMESTRUCT)
YESTERDAY_TIMESTRUCT = (date.today() - timedelta(days=1)).timetuple()
YESTERDAY_STRING = time.strftime('%Y-%m-%d', YESTERDAY_TIMESTRUCT)
BEGINNING_OF_TIME_TIMESTRUCT = time.gmtime(0)
BEGINNING_OF_TIME_STRING = time.strftime('%Y-%m-%d',
                                         BEGINNING_OF_TIME_TIMESTRUCT)


def test_build_date_arg_no_last_fetch():
    """Test build_date_arg function, with no last fetch
    """
    last_fetch_date = None
    date_arg = batch.build_date_arg(last_fetch_date)
    assert date_arg == '--start_date 1970-01-01 --end_date ' + TODAY_STRING


def test_build_date_arg_last_fetch_yesterday():
    """Test build_date_arg function, with last fetch date of yesterday
    """
    last_fetch_date = YESTERDAY_TIMESTRUCT
    date_arg = batch.build_date_arg(last_fetch_date)
    assert date_arg == '--start_date ' + YESTERDAY_STRING + \
        ' --end_date ' + TODAY_STRING


def test_maybe_run(polaris_config, tmp_path, mocker):
    """Test maybe_run function
    """
    # Disable the no-member test; this does not play well with the
    # mocked subprocess.run()
    # pylint: disable=no-member
    fullpath = tmp_path / 'maybe_run_config.json'
    with open(fullpath.as_posix(), 'w') as f_handle:
        f_handle.write(polaris_config)
    config_from_file = PolarisConfig(file=fullpath)

    my_batch_settings = {'fetch': True, 'learn': False, 'viz': True}

    config_from_file.batch_settings = my_batch_settings

    mocker.patch('subprocess.run')
    # Given the configuration above, we expect that fetch will run;
    # learn will not; and viz will run.  To test that, we call each
    # one, and check call_count() after each to make sure it's what we
    # expect.
    batch.maybe_run(cmd='fetch', config=config_from_file)
    assert subprocess.run.call_count == 1

    # Should not be called, so call count should not change...
    batch.maybe_run(cmd='learn', config=config_from_file)
    assert subprocess.run.call_count == 1

    # This *should* be called, so the call count should be 2 here.
    batch.maybe_run(cmd='viz', config=config_from_file)
    assert subprocess.run.call_count == 2
