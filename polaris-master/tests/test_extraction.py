"""
Module for testing selection.py script.
"""
import pytest
from fets.math import TSIntegrale

import polaris.learn.feature.extraction as plfe


@pytest.mark.parametrize("input_lags,transformer_class,exp_pipes", [
    (["30min", "1H", "5H"], TSIntegrale, 3),
    ([], TSIntegrale, 0),
])
def test_create_list_of_transformers(input_lags, transformer_class, exp_pipes):
    """ Testing the creating of list of transformers

        :param input_lags: see create_list_of_transformers()
        :param transformer_class: see create_list_of_transformers()
        :param exp_pipes: Expected number of pipelines
    """
    result = plfe.create_list_of_transformers(input_lags, transformer_class)
    assert len(result) == exp_pipes
