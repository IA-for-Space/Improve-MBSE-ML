"""
`pytest` testing framework file for commandline invocation
"""

import subprocess


def test_answer():
    """
    `pytest` entry point
    """

    try:
        result = subprocess.call("polaris")
        assert result == 0
    except OSError as error:
        assert error == ''
