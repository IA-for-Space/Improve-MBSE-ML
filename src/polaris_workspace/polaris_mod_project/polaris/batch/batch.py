"""Module for running polaris batch commands
"""

import datetime
import json
import logging
import subprocess
import sys
import time

from polaris.common.config import InvalidConfigurationFile, PolarisConfig

LOGGER = logging.getLogger(__name__)


def log_batch_operation(config, command, retcode):
    """Log batch operations

    :param command: command that was run
    :param retcode: exit code from that command
    """
    LOGGER.info("%s Command: %s Retcode: %d", config.name, command, retcode)


def find_last_fetch_date(config):
    """Find the date fetch was last run successfully.

    Note: we assume here that if fetch was run successfully, then we
    have all the data we need up to that point.  That is, we
    explicitly are ignoring the possibility that:

    - "polaris fetch -e 2019-12-01" was run...

    - ...it exited with exit code 0...

    - ...but for some reason, we don't have all the data from that
      day.

    :param config: polaris configuration for satellite
    :return: time of last fetch date as timetuple
    """
    normalized_frame_file = config.normalized_file_path
    LOGGER.debug('Trying to find last fetch date in %s', normalized_frame_file)
    # Copy-pasta of code in data_fetch_decoder.py.  Refactor.
    try:
        with open(normalized_frame_file) as f_handle:
            try:
                decoded_frame_list = json.load(f_handle)
            except json.JSONDecodeError:
                LOGGER.error("Cannot load % - is it a valid JSON document?",
                             normalized_frame_file)
                raise json.JSONDecodeError
            dates = [i['time'] for i in decoded_frame_list['frames']]
            latest_date = sorted(dates,
                                 key=lambda x: datetime.datetime.strptime(
                                     x, "%Y-%m-%d %H:%M:%S"))[-1]
            latest_date = datetime.datetime.strptime(latest_date,
                                                     "%Y-%m-%d %H:%M:%S")
        return latest_date.timetuple()
    except FileNotFoundError:
        return None


def build_date_arg(last_fetch_date=None):
    """Build date argument for fetch.

    :param last_fetch_date: Date of last successful fetch.
    """
    def tformat(timestamp):
        """Standard format for time arguments
        """
        return time.strftime('%Y-%m-%d', timestamp)

    if last_fetch_date is None:
        LOGGER.info('No previous fetch run for this sat, fetching everything')
        start_date = time.gmtime(0)  # Beginning of time
    else:
        start_date = last_fetch_date

    now = time.gmtime()
    return "--start_date {} --end_date {}".format(tformat(start_date),
                                                  tformat(now))


def build_fetch_args(config):
    """Build arguments for fetch command when invoked from batch.

    :param config: polaris configuration for satellite
    """
    cache_arg = '--cache_dir {}'.format(config.cache_dir)

    last_fetch_date = find_last_fetch_date(config)
    date_arg = build_date_arg(last_fetch_date=last_fetch_date)

    norm_file = config.normalized_file_path

    return '{} {} {} {}'.format(cache_arg, date_arg, config.name, norm_file)


def build_learn_args(config):
    """Build arguments for learn command when invoked from batch

    :param config: polaris configuration for satellite
    """
    args = config.normalized_file_path

    if config.learn_settings.input_file:
        args = config.learn_settings.input_file

    if config.learn_settings.configuration_file:
        args = "{} -l {}".format(args,
                                 config.learn_settings.configuration_file)

    if config.learn_settings.output_graph_file:
        args = "{} -g {}".format(args, config.learn_settings.output_graph_file)
    elif config.learn_settings.target_column is None:
        args = "{} -g {}".format(args, config.output_graph_file)

    if config.learn_settings.graph_link_threshold:
        args = "{} -t {}".format(args,
                                 config.learn_settings.graph_link_threshold)

    if config.learn_settings.target_column:
        args = "{} -c {}".format(args, config.learn_settings.target_column)

    if config.learn_settings.use_gridsearch:
        args = "{} -d".format(args)

    if config.learn_settings.force_cpu:
        args = "{} --force_cpu".format(args)

    if config.learn_settings.csv_sep:
        args = "{} -s {}".format(args, config.learn_settings.csv_sep)

    return args


def build_viz_args(config):
    """Build arguments for viz command when invoked from batch

    :param config: polaris configuration for satellite
    """
    output_graph_file = config.output_graph_file
    return '--graph_file {}'.format(output_graph_file)


def maybe_run(cmd=None, config=None, dry_run=False):
    """Run polaris command for a particular satellite

    :param cmd: command to run
    :param config: polaris configuration for satellite
    :param dry_run: bool for dry run mode
    """
    # First, check the configuration to see if we're meant to run this
    # command.
    if config.should_batch_run(cmd) is False:
        return

    LOGGER.info('Running polaris %s for %s', cmd, config.name)

    arg_builder = {}
    arg_builder['fetch'] = build_fetch_args
    arg_builder['learn'] = build_learn_args
    arg_builder['viz'] = build_viz_args

    args = arg_builder[cmd](config)
    full_cmd = 'polaris {} {}'.format(cmd, args)
    LOGGER.debug(full_cmd)
    if dry_run is True:
        return
    process_info = subprocess.run(full_cmd.split(), check=False)
    log_batch_operation(config, full_cmd, process_info.returncode)

    try:
        process_info.check_returncode()
    except subprocess.CalledProcessError:
        LOGGER.warning("%s failed", cmd)
        if config.batch_stop_at_first_failure is True:
            LOGGER.critical("Batch configured to exit on failure")
            sys.exit(1)


def batch(config_file, dry_run):
    """Run polaris fetch and learn non-interactively, based on configuration file.

    :param config_file: path to config file for batch
    :param dry_run: Bool for dry run mode
    """
    try:
        config = PolarisConfig(file=config_file)
    except FileNotFoundError:
        LOGGER.critical("Cannot find or open config file %s", config_file)
        sys.exit(1)
    except InvalidConfigurationFile:
        LOGGER.critical("Configuration file %s is invalid", config_file)
        sys.exit(1)

    for cmd in ['fetch', 'learn', 'viz']:
        maybe_run(cmd=cmd, config=config, dry_run=dry_run)
