"""
Module for testing polaris.contrib.normalizers.common.int2ddn()
"""
import pytest

import contrib.normalizers.common as contribs


@pytest.mark.parametrize("ip_as_integer,ip_as_string",
                         [(16843009, "1.1.1.1"), (16843010, "1.1.1.2"),
                          (553713922, "33.1.1.2"), (553746434, "33.1.128.2")])
def test_int2ddp(ip_as_integer, ip_as_string):
    """ Testing transformation of integers IPs representation into dotted
        decimal notation (string)

        :param ip_as_integer: integer representing a full Internet Protocol
        address
        :param ip_as_string: corresponding string representing a full Internet
        Protocol address as dotted decimal notation: 255.255.255.0
    """
    assert contribs.int2ddn(ip_as_integer) == ip_as_string
