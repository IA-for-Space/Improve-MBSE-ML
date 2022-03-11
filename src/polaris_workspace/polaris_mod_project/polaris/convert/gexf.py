"""
Converts a JSON generated graph from polaris learn
to a GEXF file format
"""
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom

from polaris.convert.graph_converter import GraphConverter

LOGGER = logging.getLogger(__name__)


class GEXFConverter(GraphConverter):
    """
    GEXFConverter converts polaris graph file to GEXF file format.
    """
    def __init__(self, graph_file_path: str, output_file_path: str):
        """ Constructor method. Note that all attributes will be initialized
            to None at the beginning and the actual conversion process does
            not happen until save_to_disk() is called.

        :param graph_file_path: Input file path for the JSON generated
        graph, including the file name and the extension
        :type graph_file_path: str
        :param output_file_path: Output file path for the generated
            GEXF file, including the file name and the extension
        :type output_file_path: str
        """
        super().__init__(graph_file_path)
        self._output_file_path = output_file_path

        self._root = None
        self._subelement_nodes = None
        self._subelement_edges = None

    def __put_nodes(self) -> None:
        """ Put all graph vertices/nodes in the nodes subelement.
        """
        for node in self.polaris_graph['graph']['nodes']:
            ET.SubElement(self._subelement_nodes,
                          'node',
                          id=node['id'],
                          label=node['name'])

    def __put_edges(self) -> None:
        """ Put all graph edges in the edges subelement.
        """
        edge_id = 0
        for edge in self.polaris_graph['graph']['links']:
            ET.SubElement(self._subelement_edges,
                          'edge',
                          id=str(edge_id),
                          source=edge['source'],
                          target=edge['target'],
                          weight=str(edge['value']))
            edge_id += 1

    def __build_subelement_edges(self) -> None:
        """ After the root element exists, we put "edges" subelement and
            fill it with edges from polaris graph.
        """
        self._subelement_edges = ET.SubElement(self._root, 'edges')
        self.__put_edges()

    def __build_subelement_nodes(self) -> None:
        """ After the root element exists, we put "nodes" subelement and
            fill it with nodes from polaris graph.
        """
        self._subelement_nodes = ET.SubElement(self._root, 'nodes')
        self.__put_nodes()

    def __build_root_element(self) -> None:
        """ Create the root element of the converted graph.
        """
        self._root = ET.Element("graph")
        self._root.set('defaultedgetype', 'directed')

    def __build_graph(self) -> None:
        """ Build an ElementTree that is a directed graph from graph output.
            Note that it is possible to have two distinct edges that point
            to the same two vertices but in opposite directions.
        """
        self.__build_root_element()
        self.__build_subelement_nodes()
        self.__build_subelement_edges()

    @staticmethod
    def get_pretty_xml(element: ET.Element) -> str:
        """ Turn the element object into a multi-line XML string.
            We have to use toprettyxml because ElementTree.tostring
            produces a long one line string.

        :param element: Any Element object, most often a root element
        :type element: ET.Element
        :return: A formatted XML string
        :rtype: str
        """
        xml_str = ET.tostring(element, encoding='utf-8', method='xml')
        # tostring produces a bytestring. We need to decode to UTF-8.
        xml_str = xml_str.decode('utf8')

        nice_xml = minidom.parseString(xml_str).toprettyxml()
        return nice_xml

    def save_to_disk(self) -> None:
        """ Write the contents of the root element (the graph)
            into a file.
        """

        self.__build_graph()
        nice_xml = self.get_pretty_xml(self._root)
        with open(self._output_file_path, 'w') as output_file:
            output_file.write(nice_xml)
