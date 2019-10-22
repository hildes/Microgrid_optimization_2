import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import xlrd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import seaborn as sns
import gurobipy
import os

create_graphml_obj = 0 # do you want to create a graph object?
create_lp_file_if_feasible_and_less_than_49_hours = 1
print_variable_values_bool = 0
create_log_file = 1

def import_planair_data():
    data_cons = pd.read_excel(
        r'/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization/Time_graph2/energy_data/Conso.xlsx')
    data_pv = pd.read_excel(
        r'/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization/Time_graph2/energy_data/PV.xlsx')
    return data_cons['kW'].values, data_pv['[kW]'].values


def print_variable_values(prob):
    print('variable values: ')
    for v in prob.variables():
        print(v.name,'=',v.varValue)


def node_hour(node):
    if node.startswith('PV'):
        return int(node[2:])
    elif node.startswith('Battery'):
        return int(node[7:])
    elif node.startswith('Consumption'):
        return int(node[11:])
    return -12


consumption_data, PV_data = import_planair_data() #in comment typical value from Christian
#value, unit
constants = {'CAPMINBAT': [0, 'kWh'],#0
             'CAPMAXBAT': [6, 'kWh'],#10
             'CAPEXVARIABLEBAT': [29, 'CHF/kWh'],#470
             'CAPEXFIXEDBAT': [3.7, 'CHF'],#3700
             'PRATEDMINSOLAR': [0, 'kWp'],#0
             'PRATEDMAXSOLAR': [15, 'kWp'],#25
             'CAPEXVARIABLESOLAR': [30, 'CHF/kWp'],#1200.0  # CHF/kWp
             'CAPEXFIXEDSOLAR': [81.0, 'CHF'],#10000.0  # fixed investment costs in CHF
             'buying_price': [0.2,'CHF'],#0.2
             'selling_price': [0.05,'CHF'],#0.05
             'PMAXINJECTEDGRID': [10,'kW'],#10
             'PMAXEXTRACTEDGRID': [10, 'kW'],#10
             'CDISCHARGEMAXBAT': [1, 'kW/kWh'],#1
             'CCHARGEMAXBAT': [1, 'kW/kWh'],#1
             'LIFETIMEBAT': [10, 'years'],#10
             'LIFETIMESOLAR': [25, 'years']#25
             }
CAPMINBAT = constants['CAPMINBAT'][0]
CAPMAXBAT = constants['CAPMAXBAT'][0]
CAPEXVARIABLEBAT = constants['CAPEXVARIABLEBAT'][0]
CAPEXFIXEDBAT = constants['CAPEXFIXEDBAT'][0]
start_hour=0
end_hour=3000
hours_considered = range(start_hour, end_hour)
PCONS = [consumption_data[i] for i in hours_considered]
NUMBER_OF_HOURS = end_hour - start_hour#len(hours_considered)
PCONS[0] = 0.0  # Otherwise the LP is infeasible !!
PCONSMAX = max(PCONS)
PRATEDSOLAR = 1  # decision variable, will take the value of the sum of PV activation variables
# production curve for installation of 1kW rated power
PNORMSOLAR = [PV_data[i] for i in hours_considered]
PRATEDMINSOLAR = constants['PRATEDMINSOLAR'][0]
PRATEDMAXSOLAR = constants['PRATEDMAXSOLAR'][0]
PGENSOLAR = np.array([PNORMSOLAR[i] for i in range(len(PNORMSOLAR))])


CAPEXVARIABLESOLAR = constants['CAPEXVARIABLESOLAR'][0]
CAPEXFIXEDSOLAR = constants['CAPEXFIXEDSOLAR'][0]
buying_price = constants['buying_price'][0]
selling_price = constants['selling_price'][0]
CENERGYGRID = np.full(NUMBER_OF_HOURS, buying_price)
CINJECTIONGRID = np.full(NUMBER_OF_HOURS, selling_price)
PMAXINJECTEDGRID = constants['PMAXINJECTEDGRID'][0]
PMAXEXTRACTEDGRID = constants['PMAXEXTRACTEDGRID'][0]
CDISCHARGEMAXBAT = constants['CDISCHARGEMAXBAT'][0]
CCHARGEMAXBAT = constants['CCHARGEMAXBAT'][0]

LIFETIMEBAT = constants['LIFETIMEBAT'][0]
LIFETIMESOLAR = constants['LIFETIMESOLAR'][0]

start_time_graph = time.time()  # start counting

G = nx.DiGraph()
# --------------adding nodes--------------
G.add_nodes_from(['supersource', 'supersink'])
pv_nodes = ['PV' + str(hour) for hour in hours_considered]
battery_nodes = ['Battery' + str(hour) for hour in hours_considered]
consumption_nodes = ['Consumption' + str(hour) for hour in hours_considered]
G.add_nodes_from(pv_nodes + battery_nodes + consumption_nodes)
# --------------adding arcs---------------
supersource_cons_edges = [('supersource', consumption_node)
                          for consumption_node in consumption_nodes]
supersource_pv_edges = [('supersource',pv_edge) for pv_edge in pv_nodes]
pv_cons_edges = [(pv_nodes[hour], consumption_nodes[hour])
                 for hour in range(NUMBER_OF_HOURS)]
pv_bat_edges = [(pv_nodes[hour], battery_nodes[hour])
                for hour in range(NUMBER_OF_HOURS)]
bat_cons_edges = [(battery_nodes[hour], consumption_nodes[hour])
                  for hour in range(NUMBER_OF_HOURS)]
bat_bat_edges = [(battery_nodes[i], battery_nodes[i + 1])
                 for i in range(NUMBER_OF_HOURS)[:-1]]
cons_supersink_edges = [(consumption_nodes[hour], 'supersink')
                        for hour in range(NUMBER_OF_HOURS)]
pv_supersink_edges = [(pv_nodes[hour], 'supersink')
                      for hour in range(NUMBER_OF_HOURS)]
G.add_edges_from(supersource_cons_edges + supersource_pv_edges + pv_cons_edges +
                 pv_bat_edges + bat_cons_edges +
                 bat_bat_edges + cons_supersink_edges + pv_supersink_edges)
print('--- %s seconds --- to create graph' % (time.time() - start_time_graph))

bat_range = range(CAPMINBAT, CAPMAXBAT + 1)
pv_range = range(PRATEDMINSOLAR, PRATEDMAXSOLAR + 1)

fixed_flow_bounds = {}
fixed_arc_flow_cost = {}
# defining fixed bounds and costs
# supsupsrcPV_supsrcPV_edges prob += flow_vars[e] <= total_sun_gen * pv_activation_vars[e]
for e in supersource_cons_edges:
    fixed_flow_bounds[e] = [0, PMAXEXTRACTEDGRID]
    fixed_arc_flow_cost[e] = CENERGYGRID[node_hour(e[1])-start_hour] #'supersource','Consumption' + str(hour)
for e in pv_cons_edges:
    fixed_flow_bounds[e] = [0,PCONSMAX]
for e in cons_supersink_edges:
    fixed_flow_bounds[e] = [PCONS[node_hour(e[0])-start_hour],PCONSMAX+1]  # upper bound could just be infinity
for e in pv_supersink_edges:
    fixed_arc_flow_cost[e]=-CINJECTIONGRID[node_hour(e[0])-start_hour]
    fixed_flow_bounds[e]=[0, PMAXINJECTEDGRID]

tab_of_objective_values = np.zeros((CAPMAXBAT-CAPMINBAT+1, PRATEDMAXSOLAR-PRATEDMINSOLAR+1))

non_zero_bat = 0
non_zero_pv = 0
for cap_bat in bat_range:
    CAPBAT = cap_bat
    if CAPBAT>0:
        non_zero_bat = 1
    for p_rated_solar in pv_range:
        start_time_loop = time.time()
        PRATEDSOLAR = p_rated_solar
        if PRATEDSOLAR >0:
            non_zero_pv = 1
        PGENSOLAR = np.array([PNORMSOLAR[i] * PRATEDSOLAR for i in
                              range(len(PNORMSOLAR))])

        flow_vars = LpVariable.dicts('flow', G.edges(), lowBound=0, cat='Continuous')
        (fixed_mins, fixed_maxs) = splitDict(fixed_flow_bounds)
        for arc_key in fixed_flow_bounds:
            flow_vars[arc_key].bounds(fixed_mins[arc_key],fixed_maxs[arc_key])

        prob = LpProblem('Energy flow problem', LpMinimize)

        # flow conservation
        for n in G.nodes():
            if not n.startswith('super'):
                prob += lpSum(flow_vars[ingoing] for ingoing in G.in_edges(n)) == \
                        lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))

        print(" --- %s seconds ---" % (time.time() - start_time_loop), ' for inner loop')
        for e in supersource_pv_edges:
            fixed_flow_bounds[e] = [0,PGENSOLAR[node_hour(e[1])]]
        for e in bat_bat_edges:
            prob += flow_vars[e] <= CAPBAT
        for e in bat_cons_edges:
             prob += flow_vars[e] <= CDISCHARGEMAXBAT * CAPBAT
        for e in pv_bat_edges:
            prob += flow_vars[e] <= CCHARGEMAXBAT * CAPBAT
        # Creates the objective function
        prob += lpSum([flow_vars[arc] * fixed_arc_flow_cost[arc] for arc in fixed_arc_flow_cost] +
                      [CAPBAT * CAPEXVARIABLEBAT,
                      PRATEDSOLAR * CAPEXVARIABLESOLAR,
                      non_zero_pv * CAPEXFIXEDSOLAR,
                      non_zero_bat * CAPEXFIXEDBAT]), "Total Cost of Energy"

        start_time = time.time()  # start counting
        prob.solve(GUROBI())
        print('battery size: ', CAPBAT, 'PV system rated power: ', PRATEDSOLAR,
              " --- %s seconds ---" % (time.time() - start_time), 'to solve')
        print(LpStatus[prob.status])
        print('Total Cost of Energy = ', value(prob.objective))
        tab_of_objective_values[cap_bat - CAPMINBAT, p_rated_solar - PRATEDMINSOLAR] = value(prob.objective)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection=Axes3D.name, xlabel='battery capacity [kW]', ylabel='PV sys. rated power [kWp]',
#                     zlabel='cost [CHF]')

#X, Y = np.meshgrid(bat_range, pv_range)
Z = tab_of_objective_values

ax = sns.heatmap(Z, linewidth=0.5,xticklabels=pv_range,yticklabels=bat_range)
plt.xlabel('rated power of PV sys. [kWp]')
plt.ylabel('battery capacity [kWh]')
plt.title('total cost of microgrid for a year [CHF] (no microgrid = 100.9 CHF)')
plt.show()

'''
grid_cons = [0]
for i in range(1, NUMBER_OF_HOURS):
    grid_cons.append(flow_vars[('supersource', 'Consumption' + str(i))].varValue)
PV_grid = [0]
for i in range(NUMBER_OF_HOURS - 1):  # last value, which is a 0, is not in the plot
    PV_grid.append(flow_vars[('PV' + str(i), 'supersink')].varValue)
battery_cons\
= [0]
for i in range(0, NUMBER_OF_HOURS - 1):
    battery_cons\
    .append(flow_vars[('Battery' + str(i), 'Consumption' + str(i + 1))].varValue)

PV_battery = [0]
for i in range(NUMBER_OF_HOURS - 1):
    PV_battery.append(flow_vars[('PV' + str(i), 'Battery' + str(i + 1))].varValue)

PV_cons = [0]
for i in range(NUMBER_OF_HOURS - 1):
    PV_cons.append(flow_vars[('PV' + str(i), 'Consumption' + str(i + 1))].varValue)

battery_usage = [0] + [flow_vars[('Battery' + str(i), 'Battery' + str(i + 1))].varValue for i in range(NUMBER_OF_HOURS-1)]

interval = range(2000,2072)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
plt.title('Energy cost: ' + str(buying_price) + ', can be sold for: ' + str(selling_price))
ax1.set_ylabel('Solar ')
ax1.plot(interval, [PV_battery[i] for i in interval], label='PV -> battery')
ax1.plot(interval, [PV_grid[i] for i in interval], label='PV -> grid', marker='+')
ax1.plot(interval, [PV_cons[i] for i in interval], label='PV -> cons', marker='x')
ax1.plot(interval, [PGENSOLAR[i] for i in interval], label='solar', color='y')

sum_PV_into_ = [PV_battery[i] + PV_grid[i] + PV_cons[i] for i in range(NUMBER_OF_HOURS)]
ax1.plot(interval, [sum_PV_into_[i] for i in interval], label='sum PV ->')

ax2.set_ylabel('consumed')
ax2.plot(interval, [PCONS[i] for i in interval], label='consumed', marker='x')
ax2.plot(interval, [grid_cons[i] for i in interval], label='grid -> cons', marker='+')
ax2.plot(interval, [battery_cons[i] for i in interval], label='battery -> cons')
ax2.plot(interval, [PV_cons[i] for i in interval], label='PV -> cons', marker='x')

ax3.plot(interval, [battery_usage[i] for i in interval])
ax3.plot(interval, np.full(len(interval), CAPBAT), marker='+')
ax3.plot(interval, [-PV_cons[i] for i in interval], label='bat -> cons',color='r')
ax3.plot(interval, [PV_battery[i] for i in interval], label = 'PV -> bat',color='g')
ax3.set_ylabel('battery usage')

leg = ax1.legend()
leg = ax2.legend()
leg = ax3.legend()
sum_into_cons = [PV_cons[i] + grid_cons[i] + battery_cons[i] for i in range(len(grid_cons))]
ax2.plot(interval, [sum_into_cons[i] for i in interval], label='into cons')
#ax2.plot(interval, PCONS[i for i in interval], label='consumption')

plt.show()
'''
# nx.write_graphml(G, '/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization/Time_graph2/test.graphml')
