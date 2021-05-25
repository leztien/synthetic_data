"""
create a random bidirectional graph (in the form of an adjacency matrix)
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def make_adjacency_matrix(n_nodes, density=None, seed=None):
    """makes a random adjacency matrix representing a graph"""
    n = n_nodes
    p = density or np.log(n_nodes)/10/n_nodes
    print(p)
    
    # Random seed
    if seed is True:
        seed = np.random.randint(1,9999)
        print("seed =", seed)
    rs = np.random.RandomState(seed)
    
    
    while True:
        mx = rs.rand(n,n)
        mx = np.triu(mx)
        mx += mx.T
        mx = (mx <= p).astype("uint8")
        mx[np.diag_indices(n)] = 0
        assert (mx == mx.T).all()
        
        
        if np.logical_or.reduce(mx, axis=0).all() and np.logical_or.reduce(mx, axis=1).all():
            break
        else:
            p += 0.01
    return(mx)


#display the graph
mx = make_adjacency_matrix(n_nodes=20, seed=True)
G = nx.from_numpy_matrix(mx)
nx.draw(G, with_labels=True)
print(mx)

##################################################################

from enum import Enum


class NodeType(Enum):
    INITIAL = 0
    GOAL = 1
    NORMAL = 2


class Node:
    def __init__(self, neighbors=None, node_id=None, node_type=None):
        self.neighbors = neighbors or []
        self.node_id = node_id
        self.node_type = node_type or NodeType.NORMAL
    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.node_id)
    def __repr__(self):
        return self.__str__()
        

def make_nodes_list(mx):
    nodes = [Node(node_id=i) for i in range(len(mx))]
    nodes[0].node_type = NodeType.INITIAL
    nodes[-1].node_type = NodeType.GOAL
    
    for row, node in zip(mx, nodes):
        node.neighbors = [nodes[ix] for ix,value in enumerate(row) if value]
    return nodes


nodes = make_nodes_list(mx)
