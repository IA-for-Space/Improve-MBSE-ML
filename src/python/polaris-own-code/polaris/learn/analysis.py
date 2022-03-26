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
                    index_column = "time",
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
        :param index_column: column to set as index of the dataframe and then drop it.
        :type index_column: str, optional
        :param regressor: name of the chosen regressor to perform the cross correlation
        :type regressor: str
        :param output_graph_file: Output file path for the generated graph.
            It will overwrite if the file already exists. Defaults to None,
            which is'/tmp/polaris_graph.json'
        :type output_graph_file: str, optional
        :param dropna: this function will perform a "drop NaN" action that will remove rows with NaN values from the dataframe.  
        :type dropna: bool, optional
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



def normalize_dataframe(dataframe, index_column="time", dropna = False):
    """
        Apply dataframe modification so it's compatible
        with the learn module. The index_column is first
        set as the index of the dataframe. Then, we drop
        the index_column.

        :param dataframe: The pandas dataframe to normalize
        :type dataframe: pd.DataFrame
        :param index_column: column to set as index of the dataframe and then drop it.
        :type index_column: str, optional
        :return: Pandas dataframe normalized
        :rtype: pd.DataFrame
        :param dropna: this function will perform a "drop NaN" action that will remove rows with NaN values from the dataframe.  
        :type dropna: bool, optional
    """
    if dropna:
        dataframe.dropna()
    dataframe.index = dataframe[index_column]
    dataframe.drop(index_column, axis=1, inplace=True)

    return dataframe

class PolarisGraphWorkbench:
    """
    To work with more than 1 graph at a time for one dataset
    """
    def __init__(self, file, index_column, delimiter_char=",", method="XGBoosting", route_to_graphs = "C:/tmp/polaris_graphs", dropna = False):
        """ Initialize an PolarisGraphWorkbench object

            :param file: The path of the dataset
            :type file: str
            :param index_column: column that will serve as index for the dataframe and then drop it
            :type index_column: str, optional
            :param delimiter_char: The character that separates the columns inside of
            the CSV file, defaults to ','
            :type delimiter_char: str, optional
            :param method: method of correlation that will be used to build the graph
            :type method: str. optional
            :param route_to_graphs: path where the jsons with the graphs will be stored
            :type route_to_graphs: str, optional
            :param dropna: if True, will remove all rows of the dataset that contain a NaN value
            :type dropna: bool, optional

        """
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
        """
        Gives instructions to visualize the graph in Polaris

        :param file: path of the file that is going to be visualized
        :type file: str 
        """
        print("Run 'polaris viz " + self.route_to_graphs + file + "' in your Python command prompt.")
        print("Visit http://localhost:8080 in your browser to see the graph.")

    def get_graph_nodes_and_links(self, file):
        """
        Returns a tuple with a dataframe of the edges of the graph and the list of all nodes
        
        :param file: path of the json where the graph is
        :type file: str 
        """
        with open(file) as f:
            graph_1 = json.loads(f.read())
        list_graph_nodes = [x["id"] for x in graph_1["graph"]["nodes"]]
        df_graph_links = pd.DataFrame([x for x in graph_1["graph"]["links"]])
        return df_graph_links, list_graph_nodes
        
    
    def get_free_nodes(self, df_links, list_nodes):
        """
        Returns the nodes of a graph which have 0 relationships 
        
        :param df_links: dataframe with all the relationships of the graph
        :type df_links: pd.DataFrame 
        :param list_nodes: list with all the nodes of the graph
        :type list_nodes: list of str 
        """
        free_nodes = []
        for node in list_nodes:
            if(node not in list(df_links["target"]) and node not in list(df_links["source"]) ):
                free_nodes.append(node)
        return free_nodes
    
    def compare_methods(self):
        """
        Displays a comparative table between the results of all graphs that has been calculated (one per correlation method)
        """
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
        """
        Runs cross_correlate with the XGBoosting regressor
        """
        cross_correlate(self.file, self.index_column, regressor="XGBoosting", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_XGBoosting.json")
        self.xgb_done = True
        print("XGBoosting graph generated")
    
    def correlate_RandomForest(self):
        """
        Runs cross_correlate with the RandomForest regressor
        """
        cross_correlate(self.file, self.index_column, regressor="RandomForest", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_randomforest.json")
        self.randomforest_done = True
        print("RandomForest graph generated")
    
    def correlate_AdaBoost(self):
        """
        Runs cross_correlate with the AdaBoost regressor
        """
        cross_correlate(self.file, self.index_column, regressor="AdaBoost", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_adaboost.json")
        self.adaboost_done = True
        print("AdaBoost graph generated")
    
    def correlate_ExtraTrees(self):
        """
        Runs cross_correlate with the ExtraTrees regressor
        """
        cross_correlate(self.file, self.index_column, regressor="ExtraTrees", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_extratrees.json")
        self.extratrees_done = True
        print("ExtraTrees graph generated")
    
    def correlate_GradientBoosting(self):
        """
        Runs cross_correlate with the GradientBoosting regressor
        """
        cross_correlate(self.file, self.index_column, regressor="GradientBoosting", dropna=self.drop_nan, csv_sep=self.delimiter_char, output_graph_file=self.route_to_graphs+"/polaris_graph_GradientBoosting.json")
        self.gradientboosting_done = True
        print("GradientBoosting graph generated")

    def XGBoosting_graph_to_dataframe(self, display_dataframe = True):
        """
        Stores the generated XGBoosting graph in a pandas DataFrame

        :param display_dataframe: if True, displays the generated dataframe
        :type display_dataframe: bool, optional
        """
        if self.xgb_done:
            self.df_XGB = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_XGBoosting.json")[0]
        else:
            self.df_XGB = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_XGB)

    def RandomForest_graph_to_dataframe(self, display_dataframe = True):
        """
        Stores the generated RandomForest graph in a pandas DataFrame

        :param display_dataframe: if True, displays the generated dataframe
        :type display_dataframe: bool, optional
        """
        if self.randomforest_done:
            self.df_RandomForest = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_RandomForest.json")[0]
        else:
            self.df_RandomForest = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_RandomForest)
            
    
    def AdaBoost_graph_to_dataframe(self, display_dataframe = True):
        """
        Stores the generated AdaBoost graph in a pandas DataFrame

        :param display_dataframe: if True, displays the generated dataframe
        :type display_dataframe: bool, optional
        """
        if self.adaboost_done:
            self.df_AdaBoost = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_AdaBoost.json")[0]
        else:
            self.df_AdaBoost = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_AdaBoost)
    
    def ExtraTrees_graph_to_dataframe(self, display_dataframe = True):
        
        """
        Stores the generated ExtraTrees graph in a pandas DataFrame

        :param display_dataframe: if True, displays the generated dataframe
        :type display_dataframe: bool, optional
        """
        if self.extratrees_done:
            self.df_ExtraTrees= self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_ExtraTrees.json")[0]
        else:
            self.df_ExtraTrees = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_ExtraTrees)
    
    def GradientBoosting_graph_to_dataframe(self, display_dataframe = True):
        """
        Stores the generated GradientBoosting graph in a pandas DataFrame

        :param display_dataframe: if True, displays the generated dataframe
        :type display_dataframe: bool, optional
        """
        if self.gradientboosting_done:
            self.df_GradientBoosting = self.get_graph_nodes_and_links(self.route_to_graphs+"/polaris_graph_GradientBoosting.json")[0]
        else:
            self.df_GradientBoosting = pd.DataFrame(columns=["source", "target", "value"])
        if display_dataframe:
            display(self.df_GradientBoosting)


    def remove_edges(self, graph, edges = []):        
        """
        Remove the indicated edges from the indicated graph

        :param graph: the graph name where the edges will be removed from 
        :type graph: str
        :param edges: the edges that will be removed from the graph
        :type edges: list of tuples (source_node, target_node)
        """
        graph_file_route = self.route_to_graphs+graph
        with open(graph_file_route) as f:
            graph_1 = json.loads(f.read())
        graph_2_links = [x for x in graph_1["graph"]["links"] if (x["source"], x["target"]) not in edges]
        graph_2 = graph_1.copy()
        graph_2["graph"]["links"] = graph_2_links

        with open(graph_file_route, 'w') as graph_file:
            graph_file.write(json.dumps(graph_2, indent = 4) )
    
    def remove_edges_from_RandomForest_graph(self, edges = []):
        """
        Remove the indicated edges from the Random Forest graph

        :param edges: the edges that will be removed from the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.randomforest_done:
            self.remove_edges("/polaris_graph_RandomForest.json", edges)
        else:
            print("RandomForest graph has not been generated")

    
    def remove_edges_from_AdaBoost_graph(self, edges = []):
        """
        Remove the indicated edges from the AdaBoost graph

        :param edges: the edges that will be removed from the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.adaboost_done:
            self.remove_edges("/polaris_graph_AdaBoost.json", edges)
        else:
            print("AdaBoost graph has not been generated")
    
    def remove_edges_from_ExtraTrees_graph(self, edges = []):
        """
        Remove the indicated edges from the Extra Trees graph

        :param edges: the edges that will be removed from the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.extratrees_done:
            self.remove_edges("/polaris_graph_ExtraTrees.json", edges)
        else:
            print("ExtraTrees graph has not been generated")

    
    def remove_edges_from_GradientBoosting_graph(self, edges = []):
        """
        Remove the indicated edges from the Gradient Boosting graph

        :param edges: the edges that will be removed from the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.gradientboosting_done:
            self.remove_edges("/polaris_graph_GradientBoosting.json", edges)
        else:
            print("GradientBoosting graph has not been generated")

    
    def remove_edges_from_XGBoosting_graph(self, edges = []):
        """
        Remove the indicated edges from the XGBoosting graph

        :param edges: the edges that will be removed from the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.xgb_done:
            self.remove_edges("/polaris_graph_XGBoosting.json", edges)
        else:
            print("XGBoosting graph has not been generated")

    def add_edges(self, graph, edges = []):
        
        """
        Add the indicated edges to the indicated graph

        :param graph: the graph name where the edges will be added to 
        :type graph: str
        :param edges: the edges that will be added to the graph
        :type edges: list of tuples (source_node, target_node)
        """
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
        """
        Add the indicated edges to the Random Forest graph

        :param edges: the edges that will be added to the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.randomforest_done:
            self.add_edges("/polaris_graph_RandomForest.json", edges)
        else:
            print("RandomForest graph has not been generated")

    
    def add_edges_to_AdaBoost_graph(self, edges = []):
        """
        Add the indicated edges to the AdaBoost graph

        :param edges: the edges that will be added to the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.adaboost_done:
            self.add_edges("/polaris_graph_AdaBoost.json", edges)
        else:
            print("AdaBoost graph has not been generated")
    
    def add_edges_to_ExtraTrees_graph(self, edges = []):
        """
        Add the indicated edges to the Estra Trees graph

        :param edges: the edges that will be added to the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.extratrees_done:
            self.add_edges("/polaris_graph_ExtraTrees.json", edges)
        else:
            print("ExtraTrees graph has not been generated")

    
    def add_edges_from_GradientBoosting_graph(self, edges = []):
        """
        Add the indicated edges to the Gradient Boosting graph

        :param edges: the edges that will be added to the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.gradientboosting_done:
            self.add_edges("/polaris_graph_GradientBoosting.json", edges)
        else:
            print("GradientBoosting graph has not been generated")

    
    def add_edges_to_XGBoosting_graph(self, edges = []):
        """
        Add the indicated edges to the XGBoosting graph

        :param edges: the edges that will be added to the graph
        :type edges: list of tuples (source_node, target_node)
        """
        if self.xgb_done:
            self.add_edges("/polaris_graph_XGBoosting.json", edges)
        else:
            print("XGBoosting graph has not been generated")
    
    def visualize_XGBoosting_graph(self):
        """
        Gives instructions to visualize the XGBoosting in Polaris

        :param file: path of the file that is going to be visualized
        :type file: str 
        """
        if self.xgb_done:
            self.visualize_graph("/polaris_graph_XGBoosting.json")
        else:
            print("XGBoosting graph has not been generated")
    
    def visualize_RandomForest_graph(self):
        """
        Gives instructions to visualize the Random Forest in Polaris

        :param file: path of the file that is going to be visualized
        :type file: str 
        """
        if self.randomforest_done:
            self.visualize_graph("/polaris_graph_randomforest.json")
        else:
            print("RandomForest graph has not been generated")
    def visualize_AdaBoost_graph(self):
        """
        Gives instructions to visualize the AdaBoost graph in Polaris

        :param file: path of the file that is going to be visualized
        :type file: str 
        """
        if self.adaboost_done:
            self.visualize_graph("/polaris_graph_adaboost.json")
        else:
            print("AdaBoost graph has not been generated")
    def visualize_ExtraTrees_graph(self):
        """
        Gives instructions to visualize the Extra Trees in Polaris

        :param file: path of the file that is going to be visualized
        :type file: str 
        """
        if self.extratrees_done:
            self.visualize_graph("/polaris_graph_extratrees.json")
        else:
            print("ExtraTrees graph has not been generated")
    def visualize_GradientBoosting_graph(self):
        """
        Gives instructions to visualize the Gradient Boosting in Polaris

        :param file: path of the file that is going to be visualized
        :type file: str 
        """
        if self.gradientboosting_done:
            self.visualize_graph("/polaris_graph_GradientBoosting.json")
        else:
            print("GradientBoosting graph has not been generated")
