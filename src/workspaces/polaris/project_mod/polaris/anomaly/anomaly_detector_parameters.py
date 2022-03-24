"""Module for AnomalyDetectorParameters class
"""


# pylint: disable=too-many-instance-attributes
# placeholder class for parameters. Hence,
# only one method and large number of parameters
class AnomalyDetectorParameters():
    """Anomaly Detector parameters class
    """
    def __init__(self):
        pass

    @property
    def window_size(self):
        """
        return window size for preprocessing
        """
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        self._window_size = window_size

    @property
    def stride(self):
        """
        return stride for preprocessing
        """
        return self._stride

    @stride.setter
    def stride(self, stride):
        self._stride = stride

    @property
    def optimizer(self):
        """
        return optimizer for predicting model like 'adam'
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def loss(self):
        """
        return loss type for predicting model like 'mean_squared_error'
        """
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def metrics(self):
        """
        return metrics for predicting model like ["MSE"]
        type: list
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def test_size_fraction(self):
        """
        return fraction of test size data in comparison to total data
        """
        return self._test_size_fraction

    @test_size_fraction.setter
    def test_size_fraction(self, test_size_fraction):
        self._test_size_fraction = test_size_fraction

    @property
    def batch_size(self):
        """
        return batch_size for predicting model
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def number_of_epochs(self):
        """
        return number_of_epochs for predicting model
        """
        return self._number_of_epochs

    @number_of_epochs.setter
    def number_of_epochs(self, number_of_epochs):
        self._number_of_epochs = number_of_epochs

    @property
    def noise_margin_per(self):
        """
        return percentage of noise above which anomaly will be detected
        """
        return self._noise_margin_per

    @noise_margin_per.setter
    def noise_margin_per(self, noise_margin_per):
        self._noise_margin_per = noise_margin_per

    @property
    def network_dimensions(self):
        """
        return network architecture of anomaly detector model
        """
        return self._network_dimensions

    @network_dimensions.setter
    def network_dimensions(self, network_dimensions):
        self._network_dimensions = network_dimensions

    @property
    def activations(self):
        """
        return activations of layers of anomaly detector model
        """
        return self._activations

    @activations.setter
    def activations(self, activations):
        self._activations = activations

    @property
    def dataset_cleaning_params(self):
        """
        Return the dataset_cleaning_params value as CleanerParameters.
        """
        return self._dataset_cleaning_params

    @dataset_cleaning_params.setter
    def dataset_cleaning_params(self, dataset_cleaning_params):
        self._dataset_cleaning_params = dataset_cleaning_params
