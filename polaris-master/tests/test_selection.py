"""
Module for testing selection.py script.
"""
from dataclasses import dataclass

import pandas as pd
import pytest
from fets.math import TSIntegrale

from polaris.learn.feature.selection import FeatureImportanceOptimization


@pytest.mark.parametrize("list_of_transformers,exp_pipes", [
    (None, 0),
    ([], 0),
    (["FAKE"], 0),
    (["FAKE", "NOT_A_TRANSFORMER"], 0),
    ([TSIntegrale("30min")], 1),
    ([(TSIntegrale("5min"), TSIntegrale("30min"))], 1),
    ([(TSIntegrale("5min"), TSIntegrale("30min")),
      TSIntegrale("15min")], 2),
])
def test_fio_init(list_of_transformers, exp_pipes):
    """ Testing the initalization of FeatureImportanceOptimization objects

        :param list_of_transformers: different list of transformers
        :param exp_pipes: Expected number of pipelines
    """
    fio = FeatureImportanceOptimization(list_of_transformers)
    assert len(fio.pipelines) == exp_pipes
    assert fio.do_tuning is False
    assert fio.model_optinput is None


@pytest.mark.parametrize(
    "list_of_fimp, method, result",
    [
        # Tests for method "first_best"
        (None, "first_best", []),
        ([[("a", 0.2), ("b", 0.8)]], "first_best", [("b", 0.8)]),
        # Test for no method provided - should behave like "first_best"
        (None, None, []),
        ([[("a", 0.2), ("b", 0.8)]], None, [("b", 0.8)]),
        # Tests for method "all_best"
        (None, "all_best", []),
        ([[("a", 0.2), ("b", 0.8)], [("d", 0.4),
                                     ("c", 0.6)]], "all_best", [("b", 0.8),
                                                                ("c", 0.6)]),
        # Tests for method "best_until_threshold"
        (None, "best_until_threshold", []),
        ([[("a", 0.01), ("aa", 0.4),
           ("b", 0.59)], [("d", 0.4),
                          ("c", 0.6)]], "best_until_threshold", [("c", 0.6),
                                                                 ("b", 0.59),
                                                                 ("aa", 0.4),
                                                                 ("d", 0.4)]),
    ])
def test_filter_importances(list_of_fimp, method, result):
    """ Test of the importance filtering

        :param list_of_fimp: list of lists of feature importances
        :param method: method for filtering
    """
    fimp_op = FeatureImportanceOptimization([TSIntegrale("30min")])
    selected_features = fimp_op.filter_importances(list_of_fimp, method)
    assert selected_features == result


@pytest.mark.parametrize("list_of_imp, result", [(None, None),
                                                 ([("a", 0.093281),
                                                   ("b", 0.069863),
                                                   ("c", 0.05984),
                                                   ("d", 0.05914),
                                                   ("e", 0.053266),
                                                   ("f", 0.01), ("g", 0.05198),
                                                   ("h", 0.05093),
                                                   ("i", 0.046125),
                                                   ("j", 0.0410098)], 4)])
def test_find_gap(list_of_imp, result):
    """ Test of finding a gap in the feature importances

        :param list_of_imp: list of feature importances
    """
    fimp_op = FeatureImportanceOptimization(list_of_imp)
    assert fimp_op.find_gap(list_of_imp) == result


@pytest.fixture(name="input_transformers")
def fixture_input_transformers():
    """ Creating a fixed set of transformers """
    transformers_list = [TSIntegrale("3H"), TSIntegrale("12H")]
    return transformers_list


def test_build_pipelines(input_transformers):
    """ Testing the function build_pipelines on the argument
        list of transformers.

        :param input_transformers: fixtures for input transformers
    """
    fio = FeatureImportanceOptimization(input_transformers)
    assert len(fio.pipelines) == 2
    # Checking if pipelines are reset
    fio.build_pipelines(input_transformers)
    assert len(fio.pipelines) == 2


def test_extract_feature_importance(input_transformers):
    """ Testing if feature importances are well extracted and ordered

        :param input_transformers: fixtures for input transformers

    """
    @dataclass
    class FakeModel:
        """ Fake Model object meant to hold the feature importance list only
        """
        feature_importances_: None

    model = FakeModel(feature_importances_=[0.5, 0.2, 0.3])

    fio = FeatureImportanceOptimization(input_transformers)
    assert len(fio.pipelines) == 2
    # Checking if importances are well listed and ordered
    result = fio.extract_feature_importance(["A", "B", "C"], model)
    assert len(result) == 3
    assert len(result[0]) == 2
    assert result[0][0] == "A"
    assert result[0][1] == 0.5
    assert result[1][0] == "C"
    assert result[1][1] == 0.3
    assert result[2][0] == "B"
    assert result[2][1] == 0.2


def test_anti_collision_renaming(input_transformers):
    """ Test if columns are well renamed during pipelined transformations

        :param input_transformers: fixtures for input transformers

    """
    fio = FeatureImportanceOptimization(input_transformers)

    # Checking renaming on Series
    series = pd.Series([3, 2, 1, "ignition"], name="myseries")
    col = "ORIGIN"
    step_n = 4

    assert series.name == "myseries"

    series = fio.anti_collision_renaming(series, col, step_n)

    assert series.name == "ORIGIN_p4_myseries"

    step_n = 5
    series = fio.anti_collision_renaming(series, col, step_n)

    assert series.name == "ORIGIN_p5_ORIGIN_p4_myseries"

    # Checking renaming on DataFrames
    dataframe = pd.DataFrame({"A": [3, 2], "B": [1, "ignition"]})

    assert len(dataframe.columns) == 2
    assert dataframe.columns[0] == "A"  # pylint: disable=E1136
    assert dataframe.columns[1] == "B"  # pylint: disable=E1136

    dataframe = fio.anti_collision_renaming(dataframe, col, step_n)

    assert len(dataframe.columns) == 2
    assert dataframe.columns[0] == "ORIGIN_p5_A"  # pylint: disable=E1136
    assert dataframe.columns[1] == "ORIGIN_p5_B"  # pylint: disable=E1136
