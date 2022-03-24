from cv2 import threshold
import pandas as pd
import warnings
from causalnex.structure import StructureModel
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas

class CausalnexDataset:

    def __init__(self, dataset, delimiter_char=",", columns_to_drop = [], threshold = 0, dropna = False):
        data = pd.read_csv(dataset, delimiter=delimiter_char)
        self._columns_to_drop = columns_to_drop
        self.data = data.drop(self._columns_to_drop, axis=1)
        if dropna == True:
            self.data = self.data.dropna()
        self.structure = from_pandas(self.data, max_iter=900)
        self.threshold = threshold
        self.edges_to_remove = []
        self.nodes_to_remove = []
        self.edges_to_add = []
        self.dropna = dropna
        

  

    def get_graph(self, specific_nodes = [], hide_specific_nodes= [], largest_subgraph = False):
        print("Run Image(result.draw(format='png')) to draw the graph")

        new_structure = self.structure.copy() 

        if len(specific_nodes) == 0:
            specific_nodes = self.structure.nodes
        else:
            df = self.edges_to_dataframe(specific_nodes)
            specific_nodes = [x for x in df["source"] if x not in hide_specific_nodes] + [x for x in df["target"] if x not in hide_specific_nodes]
        
        new_structure.remove_nodes_from([x for x in new_structure.nodes if x not in specific_nodes])

        if largest_subgraph:
            new_structure = new_structure.get_largest_subgraph()

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
                                        "fontsize": 18,
                                        "labelloc": "t",
                                        "fontcolor": "red"
                                    }
                                    for node in specific_nodes
                                }
        edge_attributes = {
                                (u, v): {
                                            "color": "white"
                                        }
                                for u, v, w in new_structure.edges(data="weight")
                                }
        viz = plot_structure(
        new_structure,
        graph_attributes=graph_attributes,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
        prog='fdp',
        )
        return viz
    
    def edges_to_dataframe(self, specific_nodes = [], hide_specific_nodes= []):
        edges = []
        

        for u,v  in self.structure.adj.items():
            for w in v:
                if len(specific_nodes) > 0:
                    if  (u in specific_nodes or w in specific_nodes) and (u not in hide_specific_nodes and w not in hide_specific_nodes)  and (u, v) and "weight" not in list(v[w]):
                        edges.append((u, w, self.threshold))
                    elif (u in specific_nodes or w in specific_nodes) and (u not in hide_specific_nodes and w not in hide_specific_nodes) and (u, v) and v[w]["weight"] >= self.threshold:
                        edges.append((u, w, v[w]["weight"]))
                else:
                    if  "weight" not in list(v[w]):
                        edges.append((u, w, self.threshold))
                    elif v[w]["weight"] >= self.threshold:
                        edges.append((u, w, v[w]["weight"]))

        df = pd.DataFrame(edges, columns=["source", "target", "weight"])
        self.edges = df
        return df
    

    def get_edges_data(self, specific_nodes=[], dataframe = None):
        if dataframe == None:
            self.edges_to_dataframe(specific_nodes)
            return self.edges.describe()
        else:
            return dataframe.describe()

    def remove_edges(self, edges=[]):
        self.edges_to_remove = edges
        adjacency = self.structure.adj.copy()
        items = adjacency.items()
        for u,v  in items:
            for w in v:
                if (u, w) in edges:
                    self.structure.remove_edge(u, w)
    

    def add_edges(self, edges=[]):
        self.edges_to_add = edges
        for edge in edges:
            self.structure.add_edge(edge[0], edge[1], origin="expert")
    
    def get_all_nodes(self):
        nodes = []
        for u,v  in self.structure.adj.items():
            nodes.append(u)
        return nodes
    
    
    def get_all_edges(self):
        edges = []
        for u,v  in self.structure.adj.items():
            for w in v:
                if "weight" not in list(v[w]):
                    edges.append((u, w, self.threshold))
                else:
                    edges.append((u, w, v[w]["weight"]))
        return edges

    def save_edges_as_dataset(self, file="graph_NOTEARS_mars_express.csv"):
        df = self.edges_to_dataframe()
        df.to_csv(file)
    
    def reset_threshold(self, threshold = 0, keep_previous_changes = True):
        self.threshold = threshold        
        self.structure = from_pandas(self.data, max_iter=900)
        if keep_previous_changes:
            self.structure.remove_edges_from(self.edges_to_remove)
            self.structure.remove_edges_below_threshold(threshold)
            self.structure.add_edges_from(self.edges_to_add)
        else:
            self.edges_to_add = []
            self.edges_to_remove = []