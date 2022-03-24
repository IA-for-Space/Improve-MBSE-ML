"""
Module for showing the list of satellites
"""

from polaris.fetch.data_fetch_decoder import _SATELLITES


def list_satellites():
    """list all the supported satellites along with the decoder/normalizer"""
    print("Supported satellites:\n")
    print(f"{'satellite'.ljust(20)}decoder/normalizer\n")
    for satellite in _SATELLITES:
        print(f"{satellite.name.ljust(20)}{satellite.decoder}")
