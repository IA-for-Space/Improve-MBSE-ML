"""
Provides testing for graph file conversion to GEXF
"""

import os
import xml.etree.ElementTree as ET
from json import JSONDecodeError

import pytest

from polaris.convert.gexf import GEXFConverter

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_convert',
)


def test_put_nodes(gexf_converter_min):
    """ Make sure all nodes are put correctly into the "nodes" subelement.

    :param gexf_converter_min: GEXF converter with minimal valid graph input
    """
    # pylint: disable=protected-access
    gexf_converter_min._GEXFConverter__build_root_element()
    gexf_converter_min._GEXFConverter__build_subelement_nodes()

    subelement_list = list(gexf_converter_min._subelement_nodes)
    assert len(subelement_list) == 3

    node_expect_list = [
        '<node id="nx_tmp" label="nx_tmp" />',
        '<node id="px_tmp" label="px_tmp" />',
        '<node id="ny_tmp" label="ny_tmp" />'
    ]

    for index, expected in enumerate(node_expect_list):
        assert ET.tostring(subelement_list[index]).decode('utf8') == expected


def test_put_edges(gexf_converter_min):
    """ Make sure all edges are put correctly into the "edges" subelement.

    :param gexf_converter_min: GEXF converter with minimal valid graph input
    """
    # pylint: disable=protected-access
    gexf_converter_min._GEXFConverter__build_root_element()
    gexf_converter_min._GEXFConverter__build_subelement_edges()

    subelement_list = list(gexf_converter_min._subelement_edges)
    assert len(subelement_list) == 3

    edge_expect_list = [
        '<edge id="0" source="nx_tmp" target="py_tmp" '
        'weight="0.5051145553588867" />',
        '<edge id="1" source="px_tmp" target="y_rate" '
        'weight="0.22684018313884735" />',
        '<edge id="2" source="ny_tmp" target="py_tmp" '
        'weight="0.14420680701732635" />'
    ]

    for index, expected in enumerate(edge_expect_list):
        assert ET.tostring(subelement_list[index]).decode('utf8') == expected


def test_create_element_structure(gexf_converter_min):
    """ Check if the root element has proper structure.
        There should be a "nodes" subelement followed
        by the "edges" subelement.

    :param gexf_converter_min: GEXF converter with minimal valid graph input
    """
    # pylint: disable=protected-access
    gexf_converter_min._GEXFConverter__build_graph()

    assert len(gexf_converter_min._root) == 2
    assert gexf_converter_min._root[0].tag == "nodes"
    assert gexf_converter_min._root[1].tag == "edges"


def test_save_pretty_xml_to_gexf_file(gexf_converter_min):
    """ Make sure the converted file is saved and
        formatted properly (not all in one line).

    :param gexf_converter_min: GEXF converter with minimal valid graph input
    """
    gexf_converter_min.save_to_disk()

    expect = [
        '<?xml version="1.0" ?>', '<graph defaultedgetype="directed">',
        '<nodes>', '<node id="nx_tmp" label="nx_tmp"/>',
        '<node id="px_tmp" label="px_tmp"/>',
        '<node id="ny_tmp" label="ny_tmp"/>', '</nodes>', '<edges>',
        '<edge id="0" source="nx_tmp" target="py_tmp" '
        'weight="0.5051145553588867"/>',
        '<edge id="1" source="px_tmp" target="y_rate" '
        'weight="0.22684018313884735"/>',
        '<edge id="2" source="ny_tmp" target="py_tmp" '
        'weight="0.14420680701732635"/>', '</edges>', '</graph>'
    ]

    # pylint: disable=protected-access
    with open(gexf_converter_min._output_file_path, 'r') as out_file:
        for index, line in enumerate(out_file):
            assert line.strip() == expect[index]


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, 'graph_ok.json'))
def test_load_graph_file_found_and_ok(datafiles):
    """ A complete file that contains graph, graph/nodes, and graph/edges key
        from polaris learn should load successfully.

    :param datafiles: Test files provided by pytest
    """
    GEXFConverter(datafiles / "graph_ok.json", "test_load_graph_out.gexf")


def test_graph_file_not_found(tmp_path):
    """ Raise FileNotFoundError if can't find the graph file.

    :param tmp_path: Temporary path provided by pytest
    """
    with pytest.raises(FileNotFoundError):
        GEXFConverter(tmp_path / "invalid_path" / "no_file.json",
                      "out_no_file.gexf")


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, 'graph_empty.json'))
def test_graph_file_exists_but_invalid(datafiles, tmp_path):
    """ Empty file (that exits) should produce a JSONDecodeError from json.load.

    :param datafiles: Test files provided by pytest
    :param tmp_path: Temporary path provided by pytest
    """
    with pytest.raises(JSONDecodeError):
        GEXFConverter(datafiles / "graph_empty.json",
                      tmp_path / "out_empty_file_graph.gexf")


@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'graph_graph_key_missing.json'))
def test_graph_file_exists_but_no_graph_key(datafiles, tmp_path):
    """ Raise KeyError if the "graph" key is not found inside the file

    :param datafiles: Test files provided by pytest
    :param tmp_path: Temporary path provided by pytest
    """
    with pytest.raises(KeyError):
        GEXFConverter(datafiles / "graph_graph_key_missing.json",
                      tmp_path / "out_no_graph.gexf")


@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'graph_nodes_key_missing.json'))
def test_graph_file_exists_but_no_nodes_key(datafiles, tmp_path):
    """ Raise KeyError if the "graph" key exists, but has
        no "nodes" key inside.

    :param datafiles: Test files provided by pytest
    :param tmp_path: Temporary path provided by pytest
    """
    with pytest.raises(KeyError):
        GEXFConverter(datafiles / "graph_nodes_key_missing.json",
                      tmp_path / "out_no_nodes.gexf")


@pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'graph_links_key_missing.json'))
def test_graph_file_exists_but_no_links_key(datafiles, tmp_path):
    """ Raise KeyError if the "graph" key exists, but has
        no "edges" key inside.

    :param datafiles: Test files provided by pytest
    :param tmp_path: Temporary path provided by pytest
    """
    with pytest.raises(KeyError):
        GEXFConverter(datafiles / "graph_links_key_missing.json",
                      tmp_path / "out_no_links.gexf")


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, 'graph_ok.json'))
def test_graph_file_convert_success(datafiles, tmp_path):
    """ Make sure a complete file that contains graph, graph/nodes,
        and graph/edges key from polaris learn convert successfully.

    :param datafiles: Test files provided by pytest
    :param tmp_path: Temporary path provided by pytest
    """
    GEXFConverter(datafiles / "graph_ok.json",
                  tmp_path / "out_ok.gexf").save_to_disk()
