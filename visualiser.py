# Note: Used for Jupyter Notebook

# from graphviz import Digraph

# def trace(root):
#     # Build a set of all nodes and edges in a graph

#     nodes = set()
#     edges = set()
    
#     def build(v):
#         if v not in nodes:
#             nodes.add(v)
#             for child in v._prev:
#                 edges.add((child, v))
#                 build(child)

#     build(root)
#     return nodes, edges

# def draw_dot(root):
#     dot = Digraph(format = "svg", graph_attr = {"rankdir": "LR"}) # LR = Left to right

#     nodes, edges = trace(root)
    
#     for n in nodes:
#         uid = str(id(n))

#         # For any value in the graph, create a rectangular ("record") node for it
#         dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.gradient), shape = "record")
        
#         if n._operation:
#             # If this value is a result of some operation, create an operation node for it
#             dot.node(name = uid + n._operation, label = n._operation)
#             # And connect this node 
#             dot.edge(uid + n._operation, uid)
    
#     for node1, node2 in edges:
#         # Connect node1 to the operation node of node2
#         dot.edge(str(id(node1)), str(id(node2)) + node2._operation)

#     return dot