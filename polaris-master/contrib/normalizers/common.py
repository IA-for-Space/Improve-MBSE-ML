""" Common contributions for normalizers
"""
import logging
from collections import namedtuple

LOGGER = logging.getLogger(__name__)

Field = namedtuple('Field', ['key', 'equ', 'unit', 'desc'])


class Normalizer:
    """ Normalizer class

        Transforms frame values according to embedded equations

    """
    def __init__(self):
        self.normalizers = []

    def normalize(self, frame):
        """ Normalize data from a frame

            :param frame: input frame
        """
        missing_keys = False

        for field in self.normalizers:
            try:
                key = field.key
                val = frame['fields'][key]
                frame['fields'][key] = {}
                frame['fields'][key]['value'] = field.equ(val)  # normalize
                frame['fields'][key]['unit'] = field.unit
            except KeyError:
                missing_keys = True
                LOGGER.debug('Field %s not found in the frame', key)

        if missing_keys:
            LOGGER.warning('Some fields could not be normalized')

        return frame

    def get_unit(self, key):
        """ Get the unit for a given key

            :param key: given key
        """
        for field in self.normalizers:
            if key == field.key:
                return field.unit
        return None

    def validate_frame(self, frame):
        """ Validate that the frame is valid for the satellite

            :param frame: input frame
        """
        # pylint: disable=no-self-use
        return frame is not None

    def get_fields_name(self):
        """ Return the names of the fields

            :return: The list of the field names
        """
        return [field.key for field in self.normalizers]


def int2ddn(val):
    """
    Convert simple integer represented IP adresses into DDN (Dotted
    Decimal Notation)

    :param val: an IP address stored as integer

    :returns out: a string containing the DDN represented IP address
    """
    out = '{}.{}.{}.{}'.format((val & 0xFF000000) >> 24,
                               (val & 0x00FF0000) >> 16,
                               (val & 0x0000FF00) >> 8, (val & 0x000000FF))
    return out
