"""
`pytest` testing framework file for cleaner configurator
"""

from polaris.feature.cleaner_configurator import CleanerConfigurator


def test_get_default_configuration():
    """
    Test for getting default configuration
    """
    configurator = CleanerConfigurator()
    parameters = configurator.get_configuration()

    assert parameters.col_max_na_percentage == 30
    assert parameters.row_max_na_percentage == 60
