"""
GraphConverter abstract class
"""
import abc
import json
import logging
from json import JSONDecodeError

LOGGER = logging.getLogger(__name__)


class GraphConverter(metaclass=abc.ABCMeta):
    """
    This is an abstract class used for every graph converter.
    Every graph converter must subclass GraphConverter and call
    super().__init__(graph_file_path) before doing any conversion.
    Every graph converter must do the conversion process given
    polaris_graph. Every conversion function must also implement
    save_to_disk function to ensure the conversion output is
    actually being saved into disk.
    """
    def __init__(self, graph_file_path: str):
        """
        Constructor method. Every polaris graph must be loaded
        and validated. Make sure polaris graph has important keys
        such as "graph", "nodes", and "links".

        :param graph_file_path: Input file path for the JSON generated
        graph, including the file name and the extension
        :type graph_file_path: str
        """
        self._graph_file_path = graph_file_path
        self.__load_graph_file()
        self.__validate_graph_file()

    def __load_graph_file(self) -> None:
        """ Read the graph file then decode it using json.load.
        """

        polaris_graph_file = open(self._graph_file_path, 'r')

        try:
            self.polaris_graph = json.load(polaris_graph_file)
        except JSONDecodeError:
            LOGGER.error("Invalid JSON file: %s", self._graph_file_path)
            raise

    def __validate_graph_file(self) -> None:
        """ Make sure important keys such as "graph", "nodes",
            and "links" exist in the input graph file.

            :raises KeyError: If there is any important key missing
                in the graph file.
        """
        if 'graph' not in self.polaris_graph:
            raise KeyError("Can't find key \"graph\" in JSON file")

        if 'nodes' not in self.polaris_graph['graph']:
            raise KeyError("Can't find key \"nodes\" inside graph")

        if 'links' not in self.polaris_graph['graph']:
            raise KeyError("Can't find key \"links\" inside graph")

    @property
    def polaris_graph(self):
        """ Return processed polaris graph output
        """
        return self._polaris_graph

    @polaris_graph.setter
    def polaris_graph(self, value):
        self._polaris_graph = value

    @abc.abstractmethod
    def save_to_disk(self):
        """ Save graph conversion result to disk.
        """
