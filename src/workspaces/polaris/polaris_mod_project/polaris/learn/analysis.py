"""
Module to launch different data analysis.
"""
import logging

from fets.math import TSIntegrale
from mlflow import set_experiment

from polaris.data.graph import PolarisGraph
from polaris.data.readers import read_polaris_data
from polaris.dataset.metadata import PolarisMetadata
from polaris.learn.feature.extraction import create_list_of_transformers, \
    extract_best_features
from polaris.learn.predictor.cross_correlation import XCorr
from polaris.learn.predictor.cross_correlation_configurator import \
    CrossCorrelationConfigurator

import os
import pandas as pd
import json
import numpy as np

LOGGER = logging.getLogger(__name__)


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
                    regressor="XGBoosting",
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
    xcorr = XCorr(metadata, xcorr_configurator.get_configuration(), regressor)
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

    def __init__(self, file, index_column, delimiter_char=",", method="XGBoosting", route_to_graphs = "C:/tmp/polaris_graphs", dropna = False):
        if method not in [ "XGBoosting", "RandomForest", "AdaBoost", "ExtraTrees" , "GradientBoosting"]:
            method = "XGBoosting"
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
        if method == "XGBoosting":
            self.correlate_XGBoosting()
        if method == "RandomForest":
            self.correlate_RandomForest()
        if method == "AdaBoost":
            self.correlate_AdaBoost()
        if method == "ExtraTrees":
            self.correlate_ExtraTrees()
        if method == "GradientBoosting":
            self.correlate_GradientBoosting()
        


    def visualize_graph(self, file):
        print("Run 'polaris viz " + self.route_to_graphs + file + "' in your Python command prompt.")
        os.system("conda polaris viz " + self.route_to_graphs + file)
        print("Visit http://localhost:8080 in your browser to see the graph.")

    def get_graph_nodes_and_links(self, file):
        with open(file) as f:
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
            df_XGB_links, list_XGB_nodes  = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_XGBoosting.json")
            num_nodes.append(len(list_XGB_nodes))
            num_edges.append(df_XGB_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_XGB_links, list_XGB_nodes)))
            average_weight.append(df_XGB_links["value"].mean())
            methods.append("XGBoosting")
        if self.randomforest_done:
            df_randomforest_links, list_randomforest_nodes = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_randomforest.json")
            num_nodes.append(len(list_randomforest_nodes))
            num_edges.append(df_randomforest_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_randomforest_links, list_randomforest_nodes)))
            average_weight.append(df_randomforest_links["value"].mean())
            methods.append("RandomForest")
        if self.adaboost_done:
            df_adaboost_links, list_adaboost_nodes = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_adaboost.json")
            num_nodes.append(len(list_adaboost_nodes))
            num_edges.append(df_adaboost_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_adaboost_links, list_adaboost_nodes)))
            average_weight.append(df_adaboost_links["value"].mean())
            methods.append("AdaBoost")
        if self.extratrees_done:
            df_extratrees_links, list_extratrees_nodes = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_extratrees.json")
            num_nodes.append(len(list_extratrees_nodes))
            num_edges.append(df_extratrees_links.shape[0])
            num_free_nodes.append(len(self.get_free_nodes(df_extratrees_links, list_extratrees_nodes)))
            average_weight.append(df_extratrees_links["value"].mean())
            methods.append("ExtraTrees")
        if self.gradientboosting_done:
            df_gradientboosting_links, list_gradientboosting_nodes = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_GradientBoosting.json")
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
        cross_correlate(self.file, self.index_column, regressor="XGBoosting", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_XGBoosting.json")
        self.xgb_done = True
        print("XGBoosting graph generated")
    
    def correlate_RandomForest(self):
        cross_correlate(self.file, self.index_column, regressor="RandomForest", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_randomforest.json")
        self.randomforest_done = True
        print("RandomForest graph generated")
    
    def correlate_AdaBoost(self):
        cross_correlate(self.file, self.index_column, regressor="AdaBoost", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_adaboost.json")
        self.adaboost_done = True
        print("AdaBoost graph generated")
    
    def correlate_ExtraTrees(self):
        cross_correlate(self.file, self.index_column, regressor="ExtraTrees", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_extratrees.json")
        self.extratrees_done = True
        print("ExtraTrees graph generated")
    
    def correlate_GradientBoosting(self):
        cross_correlate(self.file, self.index_column, regressor="GradientBoosting", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_GradientBoosting.json")
        self.gradientboosting_done = True
        print("GradientBoosting graph generated")

    def XGBoosting_graph_to_dataframe(self, display_dataframe = True):
        if self.xgb_done:
            self.df_XGB = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_XGBoosting.json")[0]
        else:
            self.df_XGB = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_XGB)

    def RandomForest_graph_to_dataframe(self, display_dataframe = True):
        if self.randomforest_done:
            self.df_RandomForest = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_RandomForest.json")[0]
        else:
            self.df_RandomForest = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_RandomForest)
            
    
    def AdaBoost_graph_to_dataframe(self, display_dataframe = True):
        if self.adaboost_done:
            self.df_AdaBoost = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_AdaBoost.json")[0]
        else:
            self.df_AdaBoost = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_AdaBoost)
    
    def ExtraTrees_graph_to_dataframe(self, display_dataframe = True):
        if self.extratrees_done:
            self.df_ExtraTrees= self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_ExtraTrees.json")[0]
        else:
            self.df_ExtraTrees = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_ExtraTrees)
    
    def GradientBoosting_graph_to_dataframe(self, display_dataframe = True):
        if self.gradientboosting_done:
            self.df_GradientBoosting = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_GradientBoosting.json")[0]
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

    
    def add_edges_to_XGBoosting_graph(self, edges = []):
        if self.xgb_done:
            self.add_edges("/polaris_graph_XGBoosting.json", edges)
        else:
            print("XGBoosting graph has not been generated")
    
    def visualize_XGBoosting_graph(self):
        if self.xgb_done:
            self.visualize_graph("/polaris_graph_XGBoosting.json")
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
