"""
Dummy normalizer class
"""
from collections import deque

from contrib.normalizers.common import Field, Normalizer


class Dummy(Normalizer):
    """
    The class providing equations for the satellite telemetry
    """
    def create_dummy_normalizer(self, field_names):
        """Creates a dummy normalizer based on field names in the list
        field_names

        :param field_names: list of field names
        :type field_names: list
        """
        normalizers = deque([])
        for field_name in field_names:
            normalizers.append(
                Field(str(field_name), lambda x: x, None, str(field_name)))
        self.normalizers = list(normalizers)
