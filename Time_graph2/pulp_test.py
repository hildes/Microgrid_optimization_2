import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from datetime import date
from datetime import datetime
import os # to create new folder

G = nx.DiGraph()

G.add_node('supersource')
G.add_node('supersink')
G.add_nodes_from(['A', 'B', 'C'])

G.add_edges_from(
    [('supersource', 'A'), ('supersource', 'B'), ('supersource', 'C'), ('A', 'C'), ('B', 'C'),
     ('A', 'supersink'),
     ('B', 'supersink'),
     ('C', 'supersink')])

# Arc : [Cost,MinFlow,MaxFlow]
arcFlowData = {('supersource', 'A'): [0, 0, 5],
               ('supersource', 'B'): [0, 0, 5],
               ('supersource', 'C'): [0, 0, 5],
               ('A', 'C'): [0, 0, 3.14],
               ('B', 'C'): [0, 0, 2.1],
               ('A', 'supersink'): [0, 0, 9],
               ('B', 'supersink'): [-1, 0, 5],
               ('C', 'supersink'): [-2, 0, 6]}

(costs, mins, maxs) = splitDict(arcFlowData)
# Creates the boundless Variables
flow_vars = LpVariable.dicts('flow', G.edges(), None, None, LpContinuous)

#for a in G.edges():
#    flow_vars[a].bounds(mins[a], maxs[a])

prob = LpProblem('Min cost flow problem', LpMinimize)

# Creates the objective function
prob += lpSum([flow_vars[a] * costs[a] for a in G.edges()]), "Total Cost of flow"
for arc_key in arcFlowData:
    prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintLE,rhs = maxs[arc_key])
    prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintGE,rhs = mins[arc_key])
# flow conservation for every node but source and sink nodes
for n in G.nodes():
    if n != 'supersource' and n != 'supersink':
        prob += lpSum(flow_vars[ingoing] for ingoing in G.in_edges(n)) == \
                lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))

#prob.writeLP('lp_files/pulp_test_LP_different_way_of_adding_constraints.lp')
prob.solve()
print(LpStatus[prob.status])

print('Total Cost of flow = ', value(prob.objective))
print('flow values:')
for v in prob.variables():
    print(v.name, "=", v.varValue)

print('selected edges: ',G.edges()-('supersource', 'A'))

H = nx.DiGraph()
node_list = ['A','B','C']
for i in range(3):
    node_list += ['A'+str(i)]
H.add_nodes_from(node_list+['supersource'])
H.add_edges_from([('supersource',x) for x in node_list])
print('edges of H: ',H.edges())

my_dict = {'v1': [1.1,'unit of 1'],
           'v2': [2.3,'unit of 2']}

v1 = my_dict['v1'][0]
print('v1 = ', v1)
