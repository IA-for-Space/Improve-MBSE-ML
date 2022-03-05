import pandas as pd
import warnings
from causalnex.structure import StructureModel
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas



def plot_graph(dataset, delimiter_char=",", columns_to_drop = [], threshold = 0):
    warnings.filterwarnings("ignore")
    data = pd.read_csv(dataset, delimiter=delimiter_char)
    structure = from_pandas(data.drop(columns_to_drop, axis=1), max_iter=900)
    graph_attributes = {
                            "splines": "spline",  # I use splies so that we have no overlap
                            "ordering": "out",
                            "ratio": "fill",  # This is necessary to control the size of the image
                            "size": "16,9!",  # Set the size of the final image.  (this is a typical presentation size)
                            "label": "",
                            "fontcolor": "#FFFFFFD9",
                            "fontname": "Helvetica",
                            "fontsize": 100,
                            "labeljust": "l",
                            "labelloc": "t",
                            "pad": "1,1",
                            "dpi": 200,
                            "nodesep": 0.8,
                            "ranksep": ".5 equally",
                            }
    node_attributes =  {
                                node: {
                                    "fontsize": 15,
                                    "labelloc": "t",
                                    "fontcolor": "red"
                                }
                                for node in structure.nodes
                            }
    edge_attributes = {
                            (u, v): {
                                        "color": "white"
                                    }
                            for u, v, w in structure.edges(data="weight")
                            }
    structure.remove_edges_below_threshold(threshold)
    viz = plot_structure(
    structure,
    graph_attributes=graph_attributes,
    node_attributes=node_attributes,
    edge_attributes=edge_attributes,
    prog='fdp',
    )
    return viz

def plot_scream(thing):
    return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

def plot_again(viz):
    Image(viz.draw(format="png"))