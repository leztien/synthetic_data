# euclidean_graph.py


"""
Generate a random eucledian graph in this format:
graph = {'vertices': {"A": (x, y), "B": (x, y)}, 'edges': {("A","B"): distance}}

and draw that graph

"""


from math import atan, degrees, sqrt, ceil
from itertools import product
import numpy as np
import matplotlib.pyplot as plt



def generate_graph(n=10, random_state=None):
    """
    Generates a random eucledian graph with n verteces and 
    approximately edge_density * (n*(n-1)/2) edges.
    Returns a dictionary in this form:
    graph = {'vertices': {"A": (x, y), "B": (x, y)}, 'edges': {("A","B"): distance}}
    """
    
    def generate_grid_points(n, height=10, width=15, random_state=None):
        """make a coordinates matrix of random points based on a grid"""
        rs = np.random.RandomState(random_state) if type(random_state) in (int, type(None)) else random_state
        side_factor = width / height
        side = sqrt(n / side_factor)
        h, w = ceil(side), ceil(side*side_factor)
        
        xx = [width / w * i for i in range(w+1)]
        yy = [height / h * i for i in range(h+1)]
        
        coordinates_matrix = np.array(list(product(xx, yy))[:n])
        coordinates_matrix += rs.normal(0, scale=width/20, size=(n,2))
        coordinates_matrix -= coordinates_matrix.min(axis=0) - 1
        coordinates_matrix = coordinates_matrix // 0.5 * 0.5
        assert len(coordinates_matrix) == n
        return coordinates_matrix
    
    # random state
    random_state = int(random_state or np.random.randint(1, 9999))
    print("random state:", random_state)
    rs = np.random.RandomState(random_state)
    
    assert n <= 26, "n must not be greater than the number of letters in the latin alphabet"
    
    # some preliminary constants
    edge_density = 0.1
    span = 10

    # this will be returned    
    graph = {'vertices': dict(), 'edges': dict()}
    
    # generate coordinates of the n points
    if rs.rand() < 0.5:
        coordinates_matrix = (rs.rand(n, 2) * span) // 0.5 * 0.5
    else:
        coordinates_matrix = generate_grid_points(n, random_state=rs)
    
    graph['vertices'] = {chr(65 + i): (x,y) for i, (x,y) in enumerate(coordinates_matrix)}
    
    # generate edges
    probabilities_matrix = rs.rand(n, n)
    probabilities_matrix[np.diag_indices(n)] = 0
    
    while edge_density < 1.0:
        adjacency_matrix = np.triu(probabilities_matrix) >= (1 - edge_density)
        edge_density += 0.001
        
        if (adjacency_matrix + adjacency_matrix.T).sum(axis=0).min() > 0:
            break
    
    for i,row in enumerate(adjacency_matrix):
        for j in np.nonzero(row)[0]:
            distance = ((coordinates_matrix[i] - coordinates_matrix[j]) ** 2).sum() ** 0.5
            graph['edges'][(chr(65 + i), chr(65 + j))] = round(distance)
    return graph

    
def draw_graph(graph, path=None):
    """Draws a geometric graph from the given graph dictionary"""
    
    # Some inner helper functions
    def plot_point(point, **kwargs):
        params = dict(marker = 'o', color='blue', zorder=0)
        params.update(kwargs)
        plt.plot(*point, **params)
        
    def annotate_point(point, text, **kwargs):
        params = dict(
            xytext=(-15, -5), textcoords='offset points',
            fontsize='x-large', fontweight='bold', zorder=1
            )
        params.update(kwargs)
        plt.annotate(text=text, xy=point, **params)
    
    def plot_line(point1, point2, **kwargs):
        params = dict(linestyle = '-', color='black', zorder=-1)
        params.update(kwargs)
        plt.plot(*zip(point1, point2), **params)
        
    def annotate_line(point1, point2, text, **kwargs):
        params = dict(xytext=(0, 10), 
                     horizontalalignment='center',
                     verticalalignment='center',
                     textcoords='offset points', fontsize='medium', color='black')
        params.update(kwargs)
        
        x, y = (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2
        slope = (point2[1] - point1[1]) / ((point2[0] - point1[0]) or 1E-9)  #to avoid devision by zero
        plt.annotate(text=text, xy=(x, y), rotation=degrees(atan(slope)), **params)
    
    # draw the grapg parts
    # plot and annotate each vertex i.e. point
    for label, point in graph['vertices'].items():
        color='green' if label=='A' else 'red' if label==chr(64+len(graph['vertices'])) else 'blue'
        plot_point(point, color=color)
        annotate_point(point, text=label, color='black')
    
    # draw and annotate each line i.e. edge
    for (point1, point2), value in graph['edges'].items():
        point1 = graph['vertices'][point1]
        point2 = graph['vertices'][point2]
        plot_line(point1, point2)
        if not path: annotate_line(point1, point2, text=value)
        
    # darw the path
    if path:
        points = [graph['vertices'][k] for k in path]
        for i in range(len(points)-1):
            plot_line(points[i], points[i+1], color='red')
            annotate_line(points[i], points[i+1], text=f"{i+1}", weight='bold')
    
    plt.axis('equal')
    return plt.gca()



if __name__ == '__main__':
    # demo 1
    graph = generate_graph(10, random_state=None)
    print(graph)
    
    draw_graph(graph)
    
    


