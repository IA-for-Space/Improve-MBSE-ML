import json
from types import SimpleNamespace

def getObjectFromGraph(graph_route):
    # Parse JSON into an object with attributes corresponding to dict keys.
    x = json.loads(graph_route, object_hook=lambda d: SimpleNamespace(**d))
    return x