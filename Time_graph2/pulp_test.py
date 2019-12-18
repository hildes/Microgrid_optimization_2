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
import os  # to create new folder

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
from jupyter_client.session import utcnow as now

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.34928, 2.45832, 2.33555, 3.95235, 0.0321, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1]])

total_cost_matrix = np.array([[3713.2, 4513.2, 4904.2, 5227.9, 5510.9, 5765.6, 6003., 6222.3, 6426.9, 6618.2,
                      6799.7],
                     [3809.5, 4609.5, 5000.1, 5316.9, 5599.1, 5855.7, 6093.6, 6311.7, 6512.1, 6700.6,
                      6882.3],
                     [3855., 4655., 5044.3, 5359.3, 5640., 5897.2, 6133.6, 6351.4, 6551., 6739.8,
                      6921.1],
                     [4023.8, 4823.8, 5207.5, 5518.1, 5795.2, 6050.5, 6284.2, 6500.3, 6698.6, 6885.6,
                      7065.6],
                     [4220.2, 5020.2, 5398.2, 5705.1, 5978.9, 6228.9, 6459.2, 6670.5, 6868.6, 7054.1,
                      7229.3],
                     [4442.4, 5242.4, 5616.3, 5918.8, 6186.4, 6429.5, 6654., 6862., 7058.2, 7240.6,
                      7411.9],
                     [4689.4, 5489.4, 5856.4, 6153., 6412., 6649., 6867.8, 7072.7, 7265.2, 7443.5,
                      7611.],
                     [4957.5, 5757.5, 6112.9, 6402., 6653.7, 6883.2, 7097.3, 7297.7, 7485.3, 7660.,
                      7821.6],
                     [5238.4, 6036.2, 6380.7, 6661.4, 6906.3, 7129.7, 7339., 7530., 7709.2, 7876.9,
                      8026.4],
                     [5525.5, 6319.9, 6654.8, 6929.9, 7165.3, 7377.1, 7575.8, 7738.5, 7840.7, 7936.2,
                      8026.4],
                     [5818.5, 6608., 6927.6, 7189.8, 7382.5, 7510.7, 7628.4, 7738.5, 7840.7, 7936.2,
                      8026.4],
                     [6107.7, 6880.5, 7081.3, 7240.7, 7382.5, 7510.7, 7628.4, 7738.5, 7840.7, 7936.2,
                      8026.4],
                     [6310.3, 6895.3, 7081.3, 7240.7, 7382.5, 7510.7, 7628.4, 7738.5, 7840.7, 7936.2,
                      8026.4]])

capex_var_solar_values = [500,700,800,900,1000,1100,1200,1300,1400,1500,1600] #  1200 typique
capex_var_bat_values = [10,20,25,45,70,100,135,175,220,270,325,385,450] #  1000 typique

fig, ax = plt.subplots()

ax1.plot([1,2,3], [2,3,1], label='PV -> cons', marker='x')

im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(total_cost_matrix[0])))
ax.set_yticks(np.arange(len(total_cost_matrix)))
# ... and label them with the respective list entries
ax.set_xticklabels(capex_var_solar_values)
ax.set_yticklabels(capex_var_bat_values)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(capex_var_bat_values)):
    for j in range(len(capex_var_solar_values)):
        text = ax.text(j, i, total_cost_matrix[i, j],
                       ha="center", va="center", color="w", fontsize=8)

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

fig1, ax1 = plt.subplots()
ax1.plot([1,2,3],[2,3,1])
plt.show()

'''
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
'''
