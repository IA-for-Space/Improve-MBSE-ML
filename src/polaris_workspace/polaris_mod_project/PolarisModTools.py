from distutils.util import execute
import os
import pandas as pd
import json
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
import xgboost as xgb
import logging
import warnings
import enlighten
import conda
import numpy as np
import pandas as pd
from polaris.feature.cleaner import Cleaner
# Used for tracking ML process results
from mlflow import log_metric, log_param, log_params, start_run
# Used for the pipeline interface of scikit learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

# eXtreme Gradient Boost algorithm
from xgboost import XGBRegressor

#RandomForest
from sklearn.ensemble import RandomForestRegressor

#Extratrees regressor
from sklearn.ensemble import ExtraTreesRegressor

#AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

#GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

from fets.math import TSIntegrale
from mlflow import set_experiment
import polaris
from polaris.data.graph import PolarisGraph
from polaris.data.readers import read_polaris_data
from polaris.dataset.metadata import PolarisMetadata
from polaris.learn.feature.extraction import create_list_of_transformers, \
    extract_best_features
from polaris.learn.predictor.cross_correlation_configurator import \
    CrossCorrelationConfigurator

LOGGER = logging.getLogger(__name__)




class XCorrMod(BaseEstimator, TransformerMixin):
    """ Cross Correlation predictor class
    """
    def __init__(self, dataset_metadata, cross_correlation_params, regressor):
        """ Initialize an XCorr object

            :param dataset_metadata: The metadata of the dataset
            :type dataset_metadata: PolarisMetadata
            :param cross_correlation_params: XCorr parameters
            :type cross_correlation_params: CrossCorrelationParameters
        """
        self._regressor = regressor
        self.models = None
        self._importances_map = None
        self._feature_cleaner = Cleaner(
            dataset_metadata, cross_correlation_params.dataset_cleaning_params)
        self.xcorr_params = {
            "random_state": cross_correlation_params.random_state,
            "test_size": cross_correlation_params.test_size,
            "gridsearch_scoring": cross_correlation_params.gridsearch_scoring,
            "gridsearch_n_splits":
            cross_correlation_params.gridsearch_n_splits,
        }
        # If we're importing from CSV, the dataset_metadata may not
        # have the feature_columns key.
        try:
            self.xcorr_params['feature_columns'] = dataset_metadata[
                'analysis']['feature_columns']
        except KeyError:
            LOGGER.info(
                "No feature_columns entry in metatdata, setting to empty array"
            )
            self.xcorr_params['feature_columns'] = []

        if cross_correlation_params.use_gridsearch:
            self.method = self.gridsearch
            self.mlf_logging = self.gridsearch_mlf_logging
        else:
            self.method = self.regression
            self.mlf_logging = self.regression_mlf_logging

        self.model_params = {
            "current": cross_correlation_params.model_params,
            "cpu": cross_correlation_params.model_cpu_params
        }
        
    @property
    def regressor(self):
        return self._regressor
    
    @regressor.setter
    def regressor(self, regressor):
        self._regressor = regressor

    @property
    def importances_map(self):
        """
        Return the importances_map value as Pandas Dataframe.

        """

        return self._importances_map

    @importances_map.setter
    def importances_map(self, importances_map):
        self._importances_map = importances_map

    def fit(self, X):
        """ Train on a dataframe

            The input dataframe will be split column by column
            considering each one as a prediction target.

            :param X: Input dataframe
            :type X: pd.DataFrame
            :raises Exception: If encountered any unhandled error
                during model fitting
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data should be a DataFrame")

        if self.models is None:
            self.models = []

        manager = enlighten.get_manager()

        LOGGER.info("Clearing Data. Removing unnecessary columns")
        X = self._feature_cleaner.drop_constant_values(X)
        X = self._feature_cleaner.drop_non_numeric_values(X)
        X = self._feature_cleaner.handle_missing_values(X)

        self.reset_importance_map(X.columns)

        parameters = self.__build_parameters(X)

        pbar = manager.counter(total=len(parameters),
                               desc="Columns",
                               unit="columns")

        with start_run(run_name='cross_correlate', nested=True):
            self.mlf_logging()
            for column in parameters:
                LOGGER.info(column)
                try:
                    self.models.append(
                        self.method(X.drop([column], axis=1), X[column],
                                    self.model_params['current']))
                except Exception as err:  # pylint: disable-msg=broad-except
                    if self.model_params['current'].get(
                            "predictor") == "gpu_predictor":
                        LOGGER.info(" ".join([
                            "Encountered error using GPU.",
                            "Trying with CPU parameters now!"
                        ]))
                        self.model_params['current'] = self.model_params['cpu']
                    else:
                        raise err
                pbar.update()

    def transform(self):
        """ Unused method in this predictor """
        return self
    
        
    def regression(self, df_in, target_series, model_params):
        """ Fit a model to predict target_series with df_in features/columns
            and retain the features importances in the dependency matrix.

            :param df_in: Input dataframe representing the context, predictors
            :type df_in: pd.DataFrame
            :param target_series: pandas series of the target variable. Share
                the same indexes as the df_in dataframe
            :type target_series: pd.Series
            :param model_params: Parameters for the XGB model
            :type model_params: dict
            :return: A fitted XGBRegressor
            :rtype: XGBRegressor
        """
        # Split df_in and target to train and test dataset
        df_in_train, df_in_test, target_train, target_test = train_test_split(
            df_in,
            target_series,
            test_size=0.2,
            random_state=self.xcorr_params['random_state'])


        regressors_dict = {"XGB": XGBRegressor(**model_params),
                           "RandomForest": RandomForestRegressor(),
                           "AdaBoost": AdaBoostRegressor(),
                           "ExtraTrees": ExtraTreesRegressor(),
                           "GradientBoosting": GradientBoostingRegressor()}

        """ if self._regressor == "XGboosting":
            # Create and train a XGBoost regressor
            regr_m = XGBRegressor(**model_params)
            
        elif self._regressor == "RandomForest":
            # Create and train a Sci-kit regressor
            regr_m = RandomForestRegressor()
        
        elif self._regressor == "AdaBoost":
            # Create and train a Sci-kit regressor
            regr_m = AdaBoostRegressor()
        
        elif self._regressor == "ExtraTrees":
             # Create and train a Sci-kit regressor
            regr_m = ExtraTreesRegressor()
        
        elif self._regressor == "GradientBoosting":
            # Create and train a Sci-kit regressor
            regr_m = GradientBoostingRegressor()
        
        elif self._regressor == "HistGradientBoosting":
            # Create and train a Sci-kit regressor
            #regr_m = HistGradientBoostingRegressor()
            pass
            
        elif self._regressor == "Voting":
            # Create and train a Sci-kit regressor
            regr_m = VotingRegressor()
        
        """
        regr_m = regressors_dict[self._regressor]
        regr_m.fit(df_in_train, target_train)

        # Make predictions
        target_series_predict = regr_m.predict(df_in_test)

        try:
            rmse = np.sqrt(
                mean_squared_error(target_test, target_series_predict))
            log_metric(target_series.name, rmse)
            LOGGER.info('Making predictions for : %s', target_series.name)
            LOGGER.info('Root Mean Square Error : %s', str(rmse))
        except Exception:  # pylint: disable-msg=broad-except
            # Because of large (close to infinite values) or nans
            LOGGER.error('Cannot find RMS Error for %s', target_series.name)
            LOGGER.debug('Expected %s, Predicted %s', str(target_test),
                         str(target_series_predict))

        # indices = np.argsort(regr_m.feature_importances_)[::-1]
        # After the model is trained
        new_row = {}
        for column, feat_imp in zip(df_in.columns,
                                    regr_m.feature_importances_):
            new_row[column] = [feat_imp]

        # Current target is not in df_in, so manually adding it
        new_row[target_series.name] = [0.0]

        # Sorting new_row to avoid concatenation warnings
        new_row = dict(sorted(new_row.items()))

        # Concatenating new information about feature importances
        if self._importances_map is not None:
            self._importances_map = pd.concat([
                self._importances_map,
                pd.DataFrame(index=[target_series.name], data=new_row)
            ])
        return regr_m

    def gridsearch(self, df_in, target_series, params):
        """ Apply grid search to fine-tune XGBoost hyperparameters
            and then call the regression method with the best grid
            search parameters.

            :param df_in: Input dataframe representing the context, predictors
            :type df_in: pd.DataFrame
            :param target_series: Pandas series of the target variable. Share
                the same indexes as the df_in dataframe
            :type target_series: pd.Series
            :param params: The hyperparameters to use on the gridsearch
                method
            :type params: dict
            :raises TypeError: If df_in is not Pandas DataFrame
            :return: A fitted XGBRegressor
            :rtype: XGBRegressor
        """
        if not isinstance(df_in, pd.DataFrame):
            LOGGER.error("Expected %s got %s for df_in in gridsearch",
                         pd.DataFrame, type(df_in))
            raise TypeError

        random_state = self.xcorr_params['random_state']
        kfolds = KFold(n_splits=self.xcorr_params['gridsearch_n_splits'],
                       shuffle=True,
                       random_state=random_state)
        regr_m = XGBRegressor(random_state=random_state,
                              predictor="cpu_predictor",
                              tree_method="auto",
                              n_jobs=-1)

        gs_regr = GridSearchCV(regr_m,
                               param_grid=params,
                               cv=kfolds,
                               scoring=self.xcorr_params['gridsearch_scoring'],
                               n_jobs=-1,
                               verbose=1)
        gs_regr.fit(df_in, target_series)

        log_param(target_series.name + ' best estimator', gs_regr.best_params_)
        LOGGER.info("%s best estimator : %s", target_series.name,
                    str(gs_regr.best_estimator_))
        return self.regression(df_in, target_series, gs_regr.best_params_)

    def reset_importance_map(self, columns):
        """
        Creating an empty importance map

        :param columns: List of column names for the importance map
        :rtype columns: pd.Index or array-like
        """
        if self._importances_map is None:
            self._importances_map = pd.DataFrame(data={}, columns=columns)

    def common_mlf_logging(self):
        """ Log the parameters used for gridsearch and regression
            in mlflow
        """
        log_param('Test size', self.xcorr_params['test_size'])
        log_param('Model', 'XGBRegressor')

    def gridsearch_mlf_logging(self):
        """ Log the parameters used for gridsearch
            in mlflow
        """
        log_param('Gridsearch scoring',
                  self.xcorr_params['gridsearch_scoring'])
        log_param('Gridsearch parameters', self.model_params)
        self.common_mlf_logging()

    def regression_mlf_logging(self):
        """ Log the parameters used for regression
            in mlflow.
        """
        self.common_mlf_logging()
        log_params(self.model_params)

    def __build_parameters(self, X):
        """ Remove features only from
            being predicted.

            :param X: The dataset
            :type X: pd.DataFrame
            :return: List of remaining features that are not removed
            :rtype: list
        """
        if self.xcorr_params['feature_columns'] is None:
            return list(X.columns)

        LOGGER.info('Removing features from the parameters : %s',
                    self.xcorr_params['feature_columns'])
        feature_to_remove = set(self.xcorr_params['feature_columns'])

        return [x for x in list(X.columns) if x not in feature_to_remove]
    
class NoFramesInInputFile(Exception):
    """Raised when frames dataframe is empty"""


def feature_extraction(input_file, param_col):
    """
    Start feature extraction using the given settings.

        :param input_file: Path of a CSV file that will be
            converted to a dataframe
        :type input_file: str
        :param param_col: Target column name
        :type param_col: str
    """
    # Create a small list of two transformers which will generate two
    # different pipelines
    transformers = create_list_of_transformers(["5min", "15min"], TSIntegrale)

    # Extract the best features of the two pipelines
    out = extract_best_features(input_file,
                                transformers,
                                target_column=param_col,
                                time_unit="ms")

    # out[0] is the FeatureImportanceOptimization object
    # from polaris.learn.feature.selection
    # pylint: disable=E1101
    print(out[0].best_features)


# pylint: disable-msg=too-many-arguments
def cross_correlate(input_file,
                    index_column,
                    regressor="XGB",
                    output_graph_file=None,
                    dropna = False,
                    xcorr_configuration_file=None,
                    graph_link_threshold=0.1,
                    use_gridsearch=False,
                    csv_sep=',',
                    force_cpu=False):
    """
    Catch linear and non-linear correlations between all columns of the
    input data.

        :param input_file: CSV or JSON file path that will be
            converted to a dataframe
        :type input_file: str
        :param output_graph_file: Output file path for the generated graph.
            It will overwrite if the file already exists. Defaults to None,
            which is'/tmp/polaris_graph.json'
        :type output_graph_file: str, optional
        :param xcorr_configuration_file: XCorr configuration file path,
            defaults to None. Refer to CrossCorrelationConfigurator for
            the default parameters
        :type xcorr_configuration_file: str, optional
        :param graph_link_threshold: Minimum link value to be considered
            as a link between two nodes
        :type graph_link_threshold: float, optional
        :param use_gridsearch: Use grid search for the cross correlation.
            If this is set to False, then it will just use regression.
            Defaults to False
        :type use_gridsearch: bool, optional
        :param csv_sep: The character that separates the columns inside of
            the CSV file, defaults to ','
        :type csv_sep: str, optional
        :param force_cpu: Force CPU for cross corelation, defaults to False
        :type force_cpu: bool, optional
        :raises NoFramesInInputFile: If there are no frames in the converted
            dataframe
    """
    # Reading input file - index is considered on first column
    metadata, dataframe = read_polaris_data(input_file, csv_sep)

    if dataframe.empty:
        LOGGER.error("Empty list of frames -- nothing to learn from!")
        raise NoFramesInInputFile

    input_data = normalize_dataframe(dataframe, index_column, dropna)
    source = metadata['satellite_name']

    set_experiment(experiment_name=source)

    xcorr_configurator = CrossCorrelationConfigurator(
        xcorr_configuration_file=xcorr_configuration_file,
        use_gridsearch=use_gridsearch,
        force_cpu=force_cpu)

    # Creating and fitting cross-correlator
    xcorr = XCorrMod(metadata, xcorr_configurator.get_configuration(), regressor)
    xcorr.fit(input_data)

    if output_graph_file is None:
        output_graph_file = "/tmp/polaris_graph_"+ regressor +".json"

    metadata = PolarisMetadata({"satellite_name": source})
    graph = PolarisGraph(metadata=metadata)
    graph.from_heatmap(xcorr.importances_map, graph_link_threshold)
    with open(output_graph_file, 'w') as graph_file:
        graph_file.write(graph.to_json())


def normalize_dataframe(dataframe, index_column, dropna = False):
    """
        Apply dataframe modification so it's compatible
        with the learn module. The time column is first
        set as the index of the dataframe. Then, we drop
        the time column.

        :param dataframe: The pandas dataframe to normalize
        :type dataframe: pd.DataFrame
        :return: Pandas dataframe normalized
        :rtype: pd.DataFrame
    """
    if dropna:
        dataframe.dropna()
    dataframe.index = dataframe[index_column]
    dataframe.drop(index_column, axis=1, inplace=True)

    return dataframe

class PolarisGraphWorkbench:

    def __init__(self, file, index_column, delimiter_char=",", method="XGB", route_to_graphs = "", dropna = False):
        self.method = method
        self.index_column = index_column
        self.file = file
        self.delimiter_char = delimiter_char
        self.route_to_graphs = route_to_graphs
        self.drop_nan = dropna
        self.cross_correlate = cross_correlate
        self.xgb_done = False
        self.randomforest_done = False
        self.adaboost_done = False
        self.extratrees_done = False
        self.gradientboosting_done = False


    def visualize_graph(self, file):
        print("To use this function you'll need Anaconda3 installed. Otherwise, run 'polaris viz " + self.route_to_graphs + file + "' in your Python command prompt.")
        #exec(open(self.route_to_graphs + "/" + method +".bat").read())
        os.system("conda polaris viz " + self.route_to_graphs + file)
        print("Visit http://localhost:8080 in your browser to see the graph")

    def count_graph_nodes_and_links(self, file):
        with open(file + ".json") as f:
            graph_1 = json.loads(f.read())
        list_graph_nodes = [x["id"] for x in graph_1["graph"]["nodes"]]
        df_graph_links = pd.DataFrame([x for x in graph_1["graph"]["links"]])
        return df_graph_links, list_graph_nodes
        
    
    def get_free_nodes(self, df_links, list_nodes):
        free_nodes = []
        for node in list_nodes:
            if(node not in list(df_links["target"]) and node not in list(df_links["source"]) ):
                free_nodes.append(node)
        return free_nodes
    
    def compare_methods(self):
        methods = []
        num_nodes = []
        num_edges = []
        num_free_nodes = []
        average_weight = []
        if self.xgb_done:
            df_XGB_links, list_XGB_nodes  = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_XGB.json")
            num_nodes.append(len(list_XGB_nodes))
            num_edges.append(df_XGB_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_XGB_links, list_XGB_nodes)))
            average_weight.append(df_XGB_links["value"].mean())
            methods.append("XGBoosting")
        if self.randomforest_done:
            df_randomforest_links, list_randomforest_nodes = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_randomforest.json")
            num_nodes.append(len(list_randomforest_nodes))
            num_edges.append(df_randomforest_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_randomforest_links, list_randomforest_nodes)))
            average_weight.append(df_randomforest_links["value"].mean())
            methods.append("RandomForest")
        if self.adaboost_done:
            df_adaboost_links, list_adaboost_nodes = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_adaboost.json")
            num_nodes.append(len(list_adaboost_nodes))
            num_edges.append(df_adaboost_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_adaboost_links, list_adaboost_nodes)))
            average_weight.append(df_adaboost_links["value"].mean())
            methods.append("AdaBoost")
        if self.extratrees_done:
            df_extratrees_links, list_extratrees_nodes = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_extratrees.json")
            num_nodes.append(len(list_extratrees_nodes))
            num_edges.append(df_extratrees_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_extratrees_links, list_extratrees_nodes)))
            average_weight.append(df_extratrees_links["value"].mean())
            methods.append("ExtraTrees")
        if self.gradientboosting_done:
            df_gradientboosting_links, list_gradientboosting_nodes = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_GradientBoosting.json")
            num_nodes.append(len(list_gradientboosting_nodes))
            num_edges.append(df_gradientboosting_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_gradientboosting_links, list_gradientboosting_nodes)))
            average_weight.append(df_gradientboosting_links["value"].mean())
            methods.append("GradientBoosting")

        todos ={"Method":methods,
                "Nodes Num": num_nodes,
                "Edges Num": num_edges,
                "Free Nodes Num": num_free_nodes,
                "Average Weight": average_weight}

        display(pd.DataFrame(todos))
    
    def correlate_XGBoosting(self):
        cross_correlate(self.file, self.index_column, regressor="XGB", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_XGB")
        self.xgb_done = True
        print("XGBoosting graph generated")
    
    def correlate_RandomForest(self):
        cross_correlate(self.file, self.index_column, regressor="RandomForest", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_randomforest")
        self.randomforest_done = True
        print("RandomForest graph generated")
    
    def correlate_AdaBoost(self):
        cross_correlate(self.file, self.index_column, regressor="AdaBoost", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_adaboost")
        self.adaboost_done = True
        print("AdaBoost graph generated")
    
    def correlate_ExtraTrees(self):
        cross_correlate(self.file, self.index_column, regressor="ExtraTrees", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_extratrees")
        self.extratrees_done = True
        print("ExtraTrees graph generated")
    
    def correlate_GradientBoosting(self):
        cross_correlate(self.file, self.index_column, regressor="GradientBoosting", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_GradientBoosting")
        self.gradientboosting_done = True
        print("GradientBoosting graph generated")

    def XGBoosting_graph_to_dataframe(self, display_dataframe = True):
        if self.xgb_done:
            self.df_XGB = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_XGB")[0]
        else:
            self.df_XGB = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_XGB)

    def RandomForest_graph_to_dataframe(self, display_dataframe = True):
        if self.randomforest_done:
            self.df_RandomForest = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_RandomForest")[0]
        else:
            self.df_RandomForest = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_RandomForest)
            
    
    def AdaBoost_graph_to_dataframe(self, display_dataframe = True):
        if self.adaboost_done:
            self.df_AdaBoost = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_AdaBoost")[0]
        else:
            self.df_AdaBoost = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_AdaBoost)
    
    def ExtraTrees_graph_to_dataframe(self, display_dataframe = True):
        if self.extratrees_done:
            self.df_ExtraTrees= self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_ExtraTrees")[0]
        else:
            self.df_ExtraTrees = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_ExtraTrees)
    
    def GradientBoosting_graph_to_dataframe(self, display_dataframe = True):
        if self.gradientboosting_done:
            self.df_GradientBoosting = self.count_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_GradientBoosting")[0]
        else:
            self.df_GradientBoosting = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_GradientBoosting)


    def remove_edges(self, graph, edges = []):
        graph_file_route = self.route_to_graphs+graph
        with open(graph_file_route) as f:
            graph_1 = json.loads(f.read())
        graph_2_links = [x for x in graph_1["graph"]["links"] if (x["source"], x["target"]) not in edges]
        graph_2 = graph_1.copy()
        graph_2["graph"]["links"] = graph_2_links

        with open(graph_file_route, 'w') as graph_file:
            graph_file.write(json.dumps(graph_2, indent = 4) )
    
    def remove_edges_from_RandomForest_graph(self, edges = []):
        if self.randomforest_done:
            self.remove_edges("/polaris_graph_RandomForest.json", edges)
        else:
            print("RandomForest graph has not been generated")

    
    def remove_edges_from_AdaBoost_graph(self, edges = []):
        if self.adaboost_done:
            self.remove_edges("/polaris_graph_AdaBoost.json", edges)
        else:
            print("AdaBoost graph has not been generated")
    
    def remove_edges_from_ExtraTrees_graph(self, edges = []):
        if self.extratrees_done:
            self.remove_edges("/polaris_graph_ExtraTrees.json", edges)
        else:
            print("ExtraTrees graph has not been generated")

    
    def remove_edges_from_GradientBoosting_graph(self, edges = []):
        if self.gradientboosting_done:
            self.remove_edges("/polaris_graph_GradientBoosting.json", edges)
        else:
            print("GradientBoosting graph has not been generated")

    
    def remove_edges_from_XGBoosting_graph(self, edges = []):
        if self.xgb_done:
            self.remove_edges("/polaris_graph_XGBoosting.json", edges)
        else:
            print("XGBoosting graph has not been generated")

    def add_edges(self, graph, edges = []):
        graph_file_route = self.route_to_graphs+graph
        with open(graph_file_route) as f:
            graph_1 = json.loads(f.read())
        all_links = graph_1["graph"]["links"]
        max_weight = np.max([x["value"] for x in all_links])
        for u, v in edges:
            link = {"source":u,"target":v, "value":max_weight}
            all_links.append(link)

        graph_2 = graph_1.copy()
        graph_2["graph"]["links"] = all_links

        with open(graph_file_route, 'w') as graph_file:
            graph_file.write(json.dumps(graph_2, indent = 4) )

    def add_edges_to_RandomForest_graph(self, edges = []):
        if self.randomforest_done:
            self.add_edges("/polaris_graph_RandomForest.json", edges)
        else:
            print("RandomForest graph has not been generated")

    
    def add_edges_to_AdaBoost_graph(self, edges = []):
        if self.adaboost_done:
            self.add_edges("/polaris_graph_AdaBoost.json", edges)
        else:
            print("AdaBoost graph has not been generated")
    
    def add_edges_to_ExtraTrees_graph(self, edges = []):
        if self.extratrees_done:
            self.add_edges("/polaris_graph_ExtraTrees.json", edges)
        else:
            print("ExtraTrees graph has not been generated")

    
    def add_edges_from_GradientBoosting_graph(self, edges = []):
        if self.gradientboosting_done:
            self.add_edges("/polaris_graph_GradientBoosting.json", edges)
        else:
            print("GradientBoosting graph has not been generated")

    
    def add_edges_to_XGB_graph(self, edges = []):
        if self.xgb_done:
            self.add_edges("/polaris_graph_XGB.json", edges)
        else:
            print("XGBoosting graph has not been generated")
    
    def visualize_XGB_graph(self):
        if self.xgb_done:
            self.visualize_graph("/polaris_graph_XGB.json")
        else:
            print("XGBoosting graph has not been generated")
    
    def visualize_RandomForest_graph(self):
        if self.randomforest_done:
            self.visualize_graph("/polaris_graph_randomforest.json")
        else:
            print("RandomForest graph has not been generated")
    def visualize_AdaBoost_graph(self):
        if self.adaboost_done:
            self.visualize_graph("/polaris_graph_adaboost.json")
        else:
            print("AdaBoost graph has not been generated")
    def visualize_ExtraTrees_graph(self):
        if self.extratrees_done:
            self.visualize_graph("/polaris_graph_extratrees.json")
        else:
            print("ExtraTrees graph has not been generated")
    def visualize_GradientBoosting_graph(self):
        if self.gradientboosting_done:
            self.visualize_graph("/polaris_graph_GradientBoosting.json")
        else:
            print("GradientBoosting graph has not been generated")
    



        