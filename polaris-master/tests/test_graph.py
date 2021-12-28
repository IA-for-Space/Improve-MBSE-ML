"""pytest framework for PolarisGraph
"""

import json

from polaris.data.graph import PolarisGraph


def test_polaris_graph_json_serializable():
    """Test that Polaris Graph is JSON-serializable
    """

    pgraph = PolarisGraph()
    exported_from_json = json.loads(pgraph.to_json())
    assert "metadata" in exported_from_json.keys()
    assert "graph" in exported_from_json.keys()
    assert "nodes" in exported_from_json['graph'].keys()
    assert "links" in exported_from_json['graph'].keys()
    assert "data_format_version" in exported_from_json['graph'].keys()


def test_polaris_graph_json_init(polaris_heatmap_fixture):
    """Test that we can set each of the keys if we want to
    """
    heatmap = polaris_heatmap_fixture['df']
    pgraph = PolarisGraph(nodes='my_nodes_key',
                          links='my_links_key',
                          target='my_target_key',
                          source='my_source_key',
                          value='my_value_key')
    pgraph.from_heatmap(heatmap)

    assert "my_nodes_key" in pgraph.graph.keys()
    assert "my_links_key" in pgraph.graph.keys()

    first_link = pgraph.graph['my_links_key'][0]
    assert "my_target_key" in first_link.keys()
    assert "my_source_key" in first_link.keys()
    assert "my_value_key" in first_link.keys()


def test_polaris_graph_from_heatmap(polaris_heatmap_fixture):
    """Test from_heatmap()
    """
    heatmap = polaris_heatmap_fixture['df']
    pgraph = PolarisGraph()
    pgraph.from_heatmap(heatmap)

    assert "nodes" in pgraph.graph.keys()
    assert "links" in pgraph.graph.keys()
    assert "data_format_version" in pgraph.graph.keys()
    assert len(pgraph.graph.keys()) == 3


def test_polaris_graph__add_nodes(polaris_heatmap_fixture):
    """Test _add_columns() method
    """
    # pylint: disable=protected-access
    heatmap = polaris_heatmap_fixture['df']
    pgraph = PolarisGraph()
    pgraph._add_nodes(heatmap)
    print(pgraph.to_json())

    nodes = pgraph.graph['nodes']
    for feature in polaris_heatmap_fixture['columns']:
        assert {"id": feature, "name": feature, "group": 0} in nodes

    assert len(nodes) == len(polaris_heatmap_fixture['columns'])


def test_polaris_graph__add_links(polaris_heatmap_fixture):
    """Test _add_columns() method
    """
    # pylint: disable=protected-access
    heatmap = polaris_heatmap_fixture['df']
    pgraph = PolarisGraph()
    pgraph._add_links(heatmap)

    links = pgraph.graph['links']
    for link in links:
        assert link['value'] != 0.1

    assert len(links) == len(polaris_heatmap_fixture['columns'])

    for expected_link in polaris_heatmap_fixture['expected_links']:
        assert expected_link in links
