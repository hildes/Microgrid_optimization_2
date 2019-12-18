import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pandas as pd
import xlrd
from datetime import date
from datetime import datetime
import gurobipy as grb
import os
import csv
import matplotlib.cm as cm
import colorsys
import seaborn as sns
from pulp import solvers

print("HELLO")
create_graphml_obj = 0  # do you want to create a graph object?
create_lp_file_if_feasible_and_less_than_49_hours = 0
print_variable_values_bool = 0
create_log_file = 1
create_csv = 1
create_plot = 1
optimize_with_gurobi = 1  # CBC is the default solver used by pulp
resolution = 1 # can be 1,1.5,2,2.5,3,4  # todo: resolution of 4/3=1.333, 6/5=1.2
averaging = False  # method of downsampling
circular_time = True
create_tendance = True

solver = ''

if optimize_with_gurobi:
    solver = 'Gurobi'
else:
    solver = 'pulp (CBC)'


def import_planair_data():
    data_cons = pd.read_excel(
        r'/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization_2/Time_graph2/energy_data/Conso2.xlsx')
    data_pv = pd.read_excel(
        r'/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization_2/Time_graph2/energy_data/PV2.xlsx')
    return data_cons['kW'].values, data_pv['kW'].values


def print_variable_values(prob):
    print('variable values: ')
    for v in prob.variables():
        print(v.name, '=', v.varValue)


def node_hour(node):
    if node.startswith('PV'):
        return int(node[2:])
    elif node.startswith('Battery'):
        return int(node[7:])
    elif node.startswith('Consumption'):
        return int(node[11:])
    return -12


consumption_data, pv_data = import_planair_data()
# value, unit
# in comment typical value from Christian
constants = {'CAP_MIN_BAT': [0.0, 'kWh'],  # 0 OK
             'CAP_MAX_BAT': [100, 'kWh'],  # 100 OK
             'CAPEX_VARIABLE_BAT': [1000.0, 'CHF/kWh'],  # 1000 OK
             'CAPEX_FIXED_BAT': [2000.0, 'CHF'],  # 2000 OK
             'OPEX_VARIABLE_BAT': [0, 'CHF/year/kWh'],  # 0 OK
             'OPEX_FIXED_BAT': [0, 'CHF/year'],  # 0 OK
             'P_RATED_MIN_SOLAR': [0, 'kWp'],  # 0 OK
             'P_RATED_MAX_SOLAR': [100, 'kWp'],  # 100 OK
             'CAPEX_VARIABLE_SOLAR': [1200.0, 'CHF/kWp'],  # 1200.0  # CHF/kWp OK
             'CAPEX_FIXED_SOLAR': [10000.0, 'CHF'],  # 10000.0  # fixed investment costs in CHF OK
             'OPEX_VARIABLE_SOLAR': [30, 'CHF/year/kWh'],  # 30 OK
             'OPEX_FIXED_SOLAR': [0, 'CHF/year'],  # 0 OK
             'BUYING_PRICE_GRID': [0.2 * resolution, 'CHF'],  # 0.2 OK
             'SELLING_PRICE_GRID': [0.05 * resolution, 'CHF'],  # 0.05 OK
             'C_POWER_GRID': [0.0, 'CHF/kW'], #0.00001 # 80 Prix de la puissance soutirée ([CHF/kW]) OK
             'PMAXINJECTEDGRID': [100, 'kW'],  # 10 OK...
             'PMAXEXTRACTEDGRID': [100, 'kW'],  # 10 OK
             'C_DISCHARGE_MAX_BAT': [1, 'kW/kWh'],  # 1 OK
             'C_CHARGE_MAX_BAT': [1, 'kW/kWh'],  # 1 OK
             'ETADISCHARGEBAT': [0.95, ' '],  # 0.95 OK
             'ETACHARGEBAT': [0.95, ' '],  # 0.95 OK
             'LIFETIMEBAT': [10, 'years'],  # 10 OK
             'LIFETIMESOLAR': [25, 'years']  # 25 OK
             }
CAP_MIN_BAT = constants['CAP_MIN_BAT'][0]
CAP_MAX_BAT = constants['CAP_MAX_BAT'][0]
CAPEX_VARIABLE_BAT = constants['CAPEX_VARIABLE_BAT'][0]
CAPEX_FIXED_BAT = constants['CAPEX_FIXED_BAT'][0]
OPEX_VARIABLE_BAT = constants['OPEX_VARIABLE_BAT'][0]
OPEX_FIXED_BAT = constants['OPEX_FIXED_BAT'][0]
start_hour = 0
end_hour = int(8760 / resolution)
hours_considered = range(start_hour, end_hour)
PCONS = [consumption_data[i]*resolution for i in hours_considered]
print([PCONS[i] for i in range(24)])
print(type(PCONS[0]))
PNORMSOLAR = [pv_data[i]*resolution * (1/28.310433) for i in hours_considered]
print('sum PNORM = ',sum(PNORMSOLAR))
print('sum conso = ', sum(PCONS))
if resolution == 1.5:
    PCONS = []
    PNORMSOLAR = []
    for i in range(int(8760 / 3)):
        PCONS += [consumption_data[3 * i] * (2 / 3) + consumption_data[3 * i + 1] * (1 / 3),
                  consumption_data[3 * i + 1] * (1 / 3)
                  + consumption_data[3 * i + 2] * 2 / 3]
        PNORMSOLAR += [pv_data[3 * i] * (2 / 3) + pv_data[3 * i + 1] * (1 / 3),
                       pv_data[3 * i + 1] * (1 / 3) + pv_data[3 * i + 2] * 2 / 3]
if resolution == 2:
    if averaging:
        PCONS = [(consumption_data[2 * i] + consumption_data[2 * i + 1]) / 2 for i in range(end_hour)]
        PNORMSOLAR = [(pv_data[2 * i] + pv_data[2 * i + 1]) / 2 for i in range(end_hour)]
    else:
        PCONS = [(consumption_data[2 * i]) for i in range(end_hour)]
        PNORMSOLAR = [pv_data[2 * i] for i in range(end_hour)]
if resolution == 2.5:
    PCONS = []
    PNORMSOLAR = []
    for i in range(int(8760 / 5)):
        PCONS += [
            consumption_data[5 * i] * (2 / 5) + consumption_data[5 * i + 1] * (2 / 5) + consumption_data[5 * i + 2] * (
                    1 / 5), consumption_data[5 * i + 2] * (1 / 5)
            + consumption_data[5 * i + 3] * 2 / 5 + consumption_data[5 * i + 4] * 2 / 5]
        PNORMSOLAR += [pv_data[5 * i] * (2 / 5) + pv_data[5 * i + 1] * (2 / 5) + pv_data[5 * i + 2] * (1 / 5),
                       pv_data[5 * i + 2] * (1 / 5)
                       + pv_data[5 * i + 3] * 2 / 5 + pv_data[5 * i + 4] * 2 / 5]
if resolution == 3:
    if averaging:
        PCONS = [(consumption_data[3 * i] + consumption_data[3 * i + 1] + consumption_data[3 * i + 2]) / 3 for i in
                 range(end_hour)]
        PNORMSOLAR = [(pv_data[3 * i] + pv_data[3 * i + 1] + pv_data[3 * i + 2]) / 3 for i in range(end_hour)]
    else:
        PCONS = [consumption_data[3 * i] for i in range(end_hour)]
        PNORMSOLAR = [pv_data[3 * i] for i in range(end_hour)]
if resolution == 4:
    if averaging:
        PCONS = [(consumption_data[4 * i] + consumption_data[4 * i + 1] + consumption_data[4 * i + 2] +
                  consumption_data[4 * i + 3]) / 4 for i in range(end_hour)]
        PNORMSOLAR = [(pv_data[4 * i] + pv_data[4 * i + 1] + pv_data[4 * i + 2] + pv_data[4 * i + 3]) / 4 for i in
                      range(end_hour)]
    else:
        PCONS = [consumption_data[4 * i] for i in range(end_hour)]
        PNORMSOLAR = [pv_data[4 * i] for i in range(end_hour)]
NUMBER_OF_HOURS = end_hour - start_hour  # len(hours_considered)

P_CONS_MAX = max(PCONS)

# production curve for installation of 1kW rated power

P_RATED_MIN_SOLAR = constants['P_RATED_MIN_SOLAR'][0]
P_RATED_MAX_SOLAR = constants['P_RATED_MAX_SOLAR'][0]
PGENSOLAR = np.array([PNORMSOLAR[i] for i in range(len(PNORMSOLAR))])

CAPEX_VARIABLE_SOLAR = constants['CAPEX_VARIABLE_SOLAR'][0]
CAPEX_FIXED_SOLAR = constants['CAPEX_FIXED_SOLAR'][0]
OPEX_VARIABLE_SOLAR = constants['OPEX_VARIABLE_SOLAR'][0]
OPEX_FIXED_SOLAR = constants['OPEX_FIXED_SOLAR'][0]
BUYING_PRICE_GRID = constants['BUYING_PRICE_GRID'][0]
SELLING_PRICE_GRID = constants['SELLING_PRICE_GRID'][0]
C_POWER_GRID = constants['C_POWER_GRID'][0]
C_ENERGY_GRID = np.full(NUMBER_OF_HOURS, BUYING_PRICE_GRID)
CINJECTIONGRID = np.full(NUMBER_OF_HOURS, SELLING_PRICE_GRID)
PMAXINJECTEDGRID = constants['PMAXINJECTEDGRID'][0]
PMAXEXTRACTEDGRID = constants['PMAXEXTRACTEDGRID'][0]
C_DISCHARGE_MAX_BAT = constants['C_DISCHARGE_MAX_BAT'][0]
C_CHARGE_MAX_BAT = constants['C_CHARGE_MAX_BAT'][0]
ETADISCHARGEBAT = constants['ETADISCHARGEBAT'][0]
ETACHARGEBAT = constants['ETACHARGEBAT'][0]

LIFETIMEBAT = constants['LIFETIMEBAT'][0]
LIFETIMESOLAR = constants['LIFETIMESOLAR'][0]

start_time_graph = time.time()  # start counting

G = nx.DiGraph()
# --------------adding nodes--------------
G.add_nodes_from(['supersource', 'supersupersourcePV', 'supersourcePV', 'supersink'])
pv_nodes = ['PV' + str(hour) for hour in hours_considered]
battery_nodes = ['Battery' + str(hour) for hour in hours_considered]
consumption_nodes = ['Consumption' + str(hour) for hour in hours_considered]
pv_bat_nodes = ['pv_bat' + str(hour) for hour in hours_considered]
bat_cons_nodes = ['bat_cons' + str(hour) for hour in hours_considered]
G.add_nodes_from(pv_nodes + battery_nodes + consumption_nodes + pv_bat_nodes + bat_cons_nodes)
# --------------adding arcs---------------
supersource_cons_edges = [('supersource', consumption_node)
                          for consumption_node in consumption_nodes]
supsrcPV_PV_edges = [('supersourcePV', pv_node) for pv_node in pv_nodes]
supsupsrcPV_supsrcPV_edge = ('supersupersourcePV', 'supersourcePV')
pv_cons_edges = [(pv_nodes[hour], consumption_nodes[hour]) for hour in range(len(hours_considered))]
pv_pv_bat_edges = [(pv_nodes[hour], pv_bat_nodes[hour]) for hour in range(len(hours_considered))]
pv_bat_bat_edges = [(pv_bat_nodes[hour], battery_nodes[hour]) for hour in range(len(hours_considered))]
bat_bat_cons_edges = [(battery_nodes[hour], bat_cons_nodes[hour]) for hour in range(len(hours_considered))]
bat_cons_cons_edges = [(bat_cons_nodes[hour], consumption_nodes[hour]) for hour in range(len(hours_considered))]
bat_bat_edges = []
if circular_time:
    bat_bat_edges = [(battery_nodes[i], battery_nodes[i + 1]) for i in range(NUMBER_OF_HOURS)[:-1]] + \
                    [(battery_nodes[len(hours_considered)-1], battery_nodes[0])]
else:
    bat_bat_edges = [(battery_nodes[i], battery_nodes[i + 1]) for i in range(NUMBER_OF_HOURS)[:-1]]

cons_supersink_edges = [(consumption_nodes[hour], 'supersink') for hour in range(NUMBER_OF_HOURS)]
pv_supersink_edges = [(pv_nodes[hour], 'supersink') for hour in range(NUMBER_OF_HOURS)]
G.add_edges_from([supsupsrcPV_supsrcPV_edge] + supersource_cons_edges + supsrcPV_PV_edges
                 + pv_cons_edges + pv_pv_bat_edges + pv_bat_bat_edges + bat_bat_cons_edges + bat_cons_cons_edges +
                 bat_bat_edges + cons_supersink_edges + pv_supersink_edges)
print('--- %s seconds --- to create graph' % (time.time() - start_time_graph))

fixed_flow_bounds = {}
fixed_arc_flow_cost = {}

for e in supersource_cons_edges:
    fixed_flow_bounds[e] = [0, PMAXEXTRACTEDGRID*resolution]
    fixed_arc_flow_cost[e] = C_ENERGY_GRID[node_hour(e[1]) - start_hour]  # 'supersource','Consumption' + str(hour)
for e in pv_cons_edges:
    fixed_flow_bounds[e] = [0, 5 * P_CONS_MAX + 99999]
for e in cons_supersink_edges:
    fixed_flow_bounds[e] = [PCONS[node_hour(e[0]) - start_hour],
                            5 * P_CONS_MAX + 99999]  # upper bound could just be infinity
for e in pv_supersink_edges:
    fixed_arc_flow_cost[e] = -CINJECTIONGRID[node_hour(e[0]) - start_hour]
    fixed_flow_bounds[e] = [0, PMAXINJECTEDGRID*resolution]

start_time_constraints = time.time()  # start counting

flow_vars = LpVariable.dicts('flow', G.edges(), 0.0, None, cat='Continuous')
prated_solar = LpVariable('rated power', P_RATED_MIN_SOLAR, P_RATED_MAX_SOLAR, cat='Continuous')
cap_bat = LpVariable('cap_bat', CAP_MIN_BAT, CAP_MAX_BAT, cat='Continuous')
non_zero_pv = LpVariable('non_zero_pv', 0, 1, cat='Binary')
non_zero_bat = LpVariable('non_zero_bat', 0, 1, cat='Binary')
max_grid_cons = LpVariable('max_grid_cons', lowBound= 0.0, upBound= P_CONS_MAX, cat='Continuous') #  weird thing:
# since there is no cost associated with max power extracted at present, the lack of upper bound makes the value
# an integer for some reason. Setting upBound= P_CONS_MAX we get good behavior from max_grid_cons. That is
# if there is a cost for max power extracted it will be the optimal max_grid_cons, if not it will be the maximum
# extracted power (=P_CONS_MAX)

prob = LpProblem('Energy flow problem', LpMinimize)

#prob.addVar(vtype=grb.GRB.CONTINUOUS, name='max_grid_cons')

# fixed investment costs if there is >0 kWp installed
(fixed_mins, fixed_maxs) = splitDict(fixed_flow_bounds)
for arc_key in fixed_flow_bounds:
    # prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintLE,rhs = fixed_maxs[arc_key])
    # prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintGE,rhs = fixed_mins[arc_key])
    # prob += flow_vars[arc_key] <= fixed_maxs[arc_key]
    # prob += fixed_mins[arc_key] <= flow_vars[arc_key]
    flow_vars[arc_key].bounds(fixed_mins[arc_key], fixed_maxs[arc_key])
for e in bat_bat_edges:
    prob += flow_vars[e] <= cap_bat, 'battery capacity' + str(e[0])
for e in pv_pv_bat_edges:
    prob += flow_vars[e] <= C_CHARGE_MAX_BAT * cap_bat, 'charging rate' + str(e[0])
for e in bat_bat_cons_edges:
    prob += flow_vars[e] <= C_DISCHARGE_MAX_BAT * cap_bat, 'discharging rate' + str(e[0])
for e in supersource_cons_edges:  # max power cost
    prob += flow_vars[e] <= max_grid_cons

total_sun_gen = sum(PNORMSOLAR)
prob += flow_vars[supsupsrcPV_supsrcPV_edge] <= total_sun_gen * prated_solar
prob += prated_solar <= non_zero_pv * P_RATED_MAX_SOLAR
# prob += non_zero_pv >= prated_solar * (1/P_RATED_MAX_SOLAR)
prob += cap_bat <= non_zero_bat * CAP_MAX_BAT
for e in supsrcPV_PV_edges:
    prob += flow_vars[e] <= PNORMSOLAR[node_hour(e[1]) - start_hour] * prated_solar
# flow conservation
for n in G.nodes():
    if not (n.startswith('super') or n.startswith('pv_bat') or n.startswith('bat_cons')) or n.startswith(
            'supersourcePV'):
        prob += lpSum(flow_vars[ingoing] for ingoing in G.in_edges(n)) == \
                lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
# loss of flow because of the battery
for n in pv_bat_nodes:
    prob += lpSum(flow_vars[ingoing] * ETACHARGEBAT for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
for n in bat_cons_nodes:
    prob += lpSum(flow_vars[ingoing] * ETADISCHARGEBAT for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
print('--- %s seconds --- to add constraints' % (time.time() - start_time_constraints))

courbe_tendance_dict = {'variable_investment_cost_solar': [],
                        'variable_investment_cost_battery': [],
                        'battery_capacity': [],
                        'rated_power': [],
                        'total_cost': []}


capex_var_solar_values = [1200]#[400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000] #  1200 typique
capex_var_bat_values = [1000]#[5,10,20,25,45,70,100,135,175,220,270,325,385,450,520] #  1000 typique

bat_capacity_matrix = np.zeros((len(capex_var_bat_values), len(capex_var_solar_values)))
rated_power_matrix = np.zeros((len(capex_var_bat_values), len(capex_var_solar_values)))
total_cost_matrix = np.zeros((len(capex_var_bat_values), len(capex_var_solar_values)))
max_power_matrix = np.zeros((len(capex_var_bat_values), len(capex_var_solar_values)))



if create_tendance:
    for solar_index, capex_var_solar in enumerate(capex_var_solar_values):
        for battery_index, capex_var_bat in enumerate(capex_var_bat_values):
            CAPEX_VARIABLE_BAT = capex_var_bat
            CAPEX_VARIABLE_SOLAR = capex_var_solar
            # Creates the objective function
            prob += lpSum([flow_vars[arc] * fixed_arc_flow_cost[arc] for arc in fixed_arc_flow_cost] +
                  [non_zero_pv * (CAPEX_FIXED_SOLAR / LIFETIMESOLAR +
                                  OPEX_FIXED_SOLAR) +
                   prated_solar * (
                           CAPEX_VARIABLE_SOLAR / LIFETIMESOLAR +
                           OPEX_VARIABLE_SOLAR)] +
                  [non_zero_bat * (CAPEX_FIXED_BAT / LIFETIMEBAT +
                                   OPEX_FIXED_BAT) +
                   cap_bat * (
                           CAPEX_VARIABLE_BAT / LIFETIMEBAT +
                           OPEX_VARIABLE_BAT)] +
                          [max_grid_cons * C_POWER_GRID])


            prob.writeLP('temporary_LP.lp')
            #m = gurobipy.read(
            #    '/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization_2/Time_graph2/temp_LP.lp')
            #m.optimize()
            start_time = time.time()  # start counting
            if optimize_with_gurobi:
                prob.solve(GUROBI())
            else:
                # prob.solve()
                prob.solve(solvers.PULP_CBC_CMD(fracGap=1.1))
            time_to_solve = time.time() - start_time
            print('--- %s seconds --- to solve' % time_to_solve)

            print('--- %s seconds --- to solve' % time_to_solve)

            print('CAPEX_VARIABLE_BAT = ',CAPEX_VARIABLE_BAT)
            print('CAPEX_VARIABLE_SOLAR = ', CAPEX_VARIABLE_SOLAR)
            print('optimized battery capacity: ', cap_bat.varValue, ' kWh')
            print('Rated power of installation = ', prated_solar.varValue, ' kWp')
            print('total cost of microgrid = ', value(prob.objective))
            #courbe_tendance_dict['variable_investment_cost_solar'] += [capex_var_solar]
            #courbe_tendance_dict['variable_investment_cost_battery'] += [capex_var_bat]
            rated_power_matrix[battery_index][solar_index] = round(prated_solar.varValue,6)
            bat_capacity_matrix[battery_index][solar_index] = round(cap_bat.varValue,1)
            total_cost_matrix[battery_index][solar_index] = round(value(prob.objective),1)
            max_power_matrix[battery_index][solar_index] = round(max_grid_cons.varValue,1)

            #rated_power_matrix[battery_index][solar_index] = 100.0
            #bat_capacity_matrix[battery_index][solar_index] = 101.0
            #total_cost_matrix[battery_index][solar_index] = 7465.2
            #max_power_matrix[battery_index][solar_index] = 16.3

    print('rated_power_matrix')
    print(rated_power_matrix)
    print('bat_capacity_matrix')
    print(bat_capacity_matrix)
    print('total_cost_matrix')
    print(total_cost_matrix)



    fig, ax = plt.subplots()
    im = ax.imshow(bat_capacity_matrix)
    ax.set_xticks(np.arange(len(capex_var_solar_values)))
    ax.set_yticks(np.arange(len(capex_var_bat_values)))
    ax.set_xticklabels(capex_var_solar_values)
    ax.set_yticklabels(capex_var_bat_values)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(capex_var_bat_values)):
        for j in range(len(capex_var_solar_values)):
            text = ax.text(j, i, bat_capacity_matrix[i, j],
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_title("Price effect on battery capacity [kWh]")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(rated_power_matrix)
    ax.set_xticks(np.arange(len(capex_var_solar_values)))
    ax.set_yticks(np.arange(len(capex_var_bat_values)))
    ax.set_xticklabels(capex_var_solar_values)
    ax.set_yticklabels(capex_var_bat_values)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(capex_var_bat_values)):
        for j in range(len(capex_var_solar_values)):
            text = ax.text(j, i, rated_power_matrix[i, j],
                           ha="center", va="center", color="w",fontsize=6)
    ax.set_title("Price effect on rated power [kWp]")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(total_cost_matrix)
    ax.set_xticks(np.arange(len(capex_var_solar_values)))
    ax.set_yticks(np.arange(len(capex_var_bat_values)))
    ax.set_xticklabels(capex_var_solar_values)
    ax.set_yticklabels(capex_var_bat_values)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(capex_var_bat_values)):
        for j in range(len(capex_var_solar_values)):
            text = ax.text(j, i, total_cost_matrix[i, j],
                           ha="center", va="center", color="w",fontsize=5)
    ax.set_title("Price effect on total annual cost [CHF]")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(max_power_matrix)
    ax.set_xticks(np.arange(len(capex_var_solar_values)))
    ax.set_yticks(np.arange(len(capex_var_bat_values)))
    ax.set_xticklabels(capex_var_solar_values)
    ax.set_yticklabels(capex_var_bat_values)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(capex_var_bat_values)):
        for j in range(len(capex_var_solar_values)):
            text = ax.text(j, i, max_power_matrix[i, j],
                           ha="center", va="center", color="w",fontsize = 6)
    ax.set_title("Price effect on maximum power extracted [kW]")
    fig.tight_layout()
    plt.show()

if not create_tendance:
    prob += lpSum([flow_vars[arc] * fixed_arc_flow_cost[arc] for arc in fixed_arc_flow_cost] +
                  [non_zero_pv * (CAPEX_FIXED_SOLAR / LIFETIMESOLAR +
                                  OPEX_FIXED_SOLAR) +
                   prated_solar * (
                           CAPEX_VARIABLE_SOLAR / LIFETIMESOLAR +
                           OPEX_VARIABLE_SOLAR)] +
                  [non_zero_bat * (CAPEX_FIXED_BAT / LIFETIMEBAT +
                                   OPEX_FIXED_BAT) +
                   cap_bat * (
                           CAPEX_VARIABLE_BAT / LIFETIMEBAT +
                           OPEX_VARIABLE_BAT)] +
                  [max_grid_cons * C_POWER_GRID])
    start_time = time.time()  # start counting
    if optimize_with_gurobi:
        prob.solve(GUROBI())
    else:
        #prob.solve()
        prob.solve(solvers.PULP_CBC_CMD(fracGap=0.1))
    time_to_solve = time.time() - start_time
    print('--- %s seconds --- to solve' % time_to_solve)


if LpStatus[prob.status] == 'Optimal' and create_lp_file_if_feasible_and_less_than_49_hours and NUMBER_OF_HOURS < 49:
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hh%M")
    prob.writeLP('lp_files/' + dt_string + '_LP.lp')

print('total Cost of microgrid = ', value(prob.objective))
print('total theoretical cost without microgrid: ' + str(sum([C_ENERGY_GRID[hour] * PCONS[hour] for hour in
                                                              range(len(
                                                                  hours_considered))]) + P_CONS_MAX * C_POWER_GRID) + ' CHF')
if print_variable_values_bool == 1:
    print_variable_values(prob)
print('optimized battery capacity: ', cap_bat.varValue, ' kWh')

print('Rated power of installation = ', prated_solar.varValue, ' kWp')

''' # if infeasible, this points to an impossible constraint
print('The model is infeasible; computing IIS')
prob.writeLP('infeasible_LP.lp')
m = gurobipy.read('/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization_2/Time_graph2/infeasible_LP.lp')
m.computeIIS()
if m.IISMinimal:
  print('IIS is minimal\n')
else:
  print('IIS is not minimal\n')
print('\nThe following constraint(s) cannot be satisfied:')
for c in m.getConstrs():
    if c.IISConstr:
        print('%s' % c.constrName)
'''

sub_interval = list(range(end_hour-48, end_hour))
sub_interval = list(range(start_hour, start_hour+48))
hours_considered_set = set(hours_considered)
interval = list(hours_considered_set.intersection(sub_interval))
interval_indices = range(len(sub_interval))
interval_indices = sub_interval
interval = sub_interval

data_dict = {}

data_dict['index'] = list(range(NUMBER_OF_HOURS))

grid_cons = [flow_vars[supersource_cons_edge].varValue for supersource_cons_edge in supersource_cons_edges]
data_dict['grid_cons'] = grid_cons

tot_energy_bought = np.concatenate([[sum(grid_cons)],np.full(NUMBER_OF_HOURS-1,0)])
data_dict['tot_energy_bought'] = tot_energy_bought

pv_grid = [flow_vars[pv_supersink_edge].varValue for pv_supersink_edge in pv_supersink_edges]
data_dict['pv_grid'] = pv_grid

tot_energy_sold = np.concatenate([[sum(pv_grid)],np.full(NUMBER_OF_HOURS-1,0)])
data_dict['tot_energy_sold'] = tot_energy_sold

data_dict['tot_energy_produced_pv'] = np.concatenate([[sum(PNORMSOLAR) * prated_solar.varValue],np.full(NUMBER_OF_HOURS-1,0)])


data_dict['rated_power'] = np.concatenate([[prated_solar.varValue],np.full(NUMBER_OF_HOURS-1,0)])
data_dict['battery_capacity'] = np.concatenate([[cap_bat.varValue],np.full(NUMBER_OF_HOURS-1,0)])
data_dict['max_power_bought'] = np.concatenate([[max_grid_cons.varValue],np.full(NUMBER_OF_HOURS-1,0)])
revenue_from_selling_energy = sum(pv_grid) * SELLING_PRICE_GRID
data_dict['revenue_from_selling_energy'] = np.concatenate([[revenue_from_selling_energy],np.full(NUMBER_OF_HOURS-1,0)])
data_dict['total_energy_consumed'] = np.concatenate([[sum(PCONS)],np.full(NUMBER_OF_HOURS-1,0)])
pv_cons = [flow_vars[pv_cons_edge].varValue for pv_cons_edge in pv_cons_edges]
data_dict['pv_cons'] = pv_cons
data_dict['sum_pv_cons'] = np.concatenate([[sum(pv_cons)],np.full(NUMBER_OF_HOURS-1,0)])
battery_cons = [flow_vars[e].varValue for e in bat_cons_cons_edges]
data_dict['battery_cons'] = battery_cons
data_dict['sum_bat_cons_cons'] = np.concatenate([[sum(battery_cons)],np.full(NUMBER_OF_HOURS-1,0)])
TOT_OPEX_SOLAR = non_zero_pv.varValue * OPEX_FIXED_SOLAR + prated_solar.varValue * OPEX_VARIABLE_SOLAR
data_dict['TOT_OPEX_SOLAR'] = np.concatenate([[TOT_OPEX_SOLAR],np.full(NUMBER_OF_HOURS-1,0)])
investment_solar = non_zero_pv.varValue * (CAPEX_FIXED_SOLAR / 1.0)+prated_solar.varValue * (
                           CAPEX_VARIABLE_SOLAR / 1.0)
data_dict['investment_solar'] = np.concatenate([[investment_solar],np.full(NUMBER_OF_HOURS-1,0)])

lost_discharging = [flow_vars[e].varValue * (1 - ETADISCHARGEBAT) for e in bat_cons_cons_edges]
data_dict['lost_discharging'] = lost_discharging
pv_battery = [flow_vars[e].varValue for e in pv_pv_bat_edges]
data_dict['pv_battery'] = pv_battery
lost_charging = [flow_vars[e].varValue * (1 - ETACHARGEBAT) for e in pv_pv_bat_edges]
data_dict['lost_charging'] = lost_charging

battery_usage = [flow_vars[e].varValue for e in bat_bat_edges]
if not circular_time:
    battery_usage = [0]+battery_usage
data_dict['battery_usage'] = battery_usage
data_dict['consumption_data'] = consumption_data
data_dict['normalized_pv'] = pv_data

data_dict['solver'] = np.concatenate([[solver],np.full(NUMBER_OF_HOURS-1,0)])
data_dict['circular_time'] = np.concatenate([[circular_time],np.full(NUMBER_OF_HOURS-1,0)])

c = 0
for h in range(len(hours_considered)):
    if pv_battery[h] > 0 and battery_cons[h] > 0:
        print(h, ' charging and discharging at the same time')
        c += 1
if c == 0:
    print('never charging and discharging at the same time')
print('max_grid_cons * C_POWER_GRID = ', max_grid_cons.varValue, '*', C_POWER_GRID, '=',
      max_grid_cons.varValue * C_POWER_GRID)

pltsize = 0.48
pltratio = 0.35
fontsize = 12
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(
    len(interval) * pltsize, len(interval) * pltsize * pltratio))
ax1.title.set_text(
    'Rated power of PV installation = ' + str(prated_solar) + ' kWp, total cost: ' + str(value(prob.objective))[
                                                                                     0:12] + ' CHF')

ax1.set_ylabel('Solar ', fontsize=fontsize)
ax1.plot(sub_interval, [pv_battery[i] for i in interval_indices], label='PV -> battery')
ax1.plot(sub_interval, [pv_grid[i]
                        for i in interval_indices], label='PV -> grid', marker='+')
ax1.plot(sub_interval, [pv_cons[i]
                        for i in interval_indices], label='PV -> cons', marker='x')

ax1.plot(sub_interval, [PGENSOLAR[i] for i in interval_indices], label='normalized solar curve', color='y')
sum_PV_into_ = [pv_battery[i] + pv_grid[i] + pv_cons[i]
                for i in range(NUMBER_OF_HOURS)]
ax1.plot(sub_interval, [sum_PV_into_[i] for i in interval_indices], label='sum PV ->')

ax2.plot(sub_interval, [battery_cons[i]
                        for i in interval_indices], label='battery -> cons')
ax2.plot(sub_interval, [pv_cons[i]
                        for i in interval_indices], label='PV -> cons', marker='x')
ax2.set_ylabel('consumption', fontsize=fontsize)
ax2.plot(sub_interval, [PCONS[i]
                        for i in interval_indices], label='consumption', marker='x')
ax2.plot(sub_interval, [grid_cons[i]
                        for i in interval_indices], label='grid -> cons', marker='+')
sum_into_cons = [pv_cons[i] + grid_cons[i] + battery_cons[i]
                 for i in range(len(grid_cons))]
ax2.plot(sub_interval, [sum_into_cons[i] for i in interval_indices], label='into cons')
ax2.title.set_text('Energy cost: ' + str(BUYING_PRICE_GRID) +
                   ', can be sold for: ' + str(SELLING_PRICE_GRID))

ax3.plot(sub_interval, [-battery_cons[i]
                        for i in interval_indices], label='bat -> cons', color='r')
ax3.plot(sub_interval, [pv_battery[i]
                        for i in interval_indices], label='PV -> bat', color='g')
ax3.plot(sub_interval, np.full(len(interval_indices), cap_bat.varValue), label='bat capacity')
ax3.set_ylabel('battery usage', fontsize=fontsize)
ax3.set_xlabel('hours', fontsize=fontsize)
ax3.plot(sub_interval, [battery_usage[i] for i in interval_indices], label='bat usage')
#ax3.plot(list(list(range(end_hour-15, end_hour))+list(range(24))), [battery_usage[i] for i in interval_indices]+
#         [flow_vars[(battery_nodes[NUMBER_OF_HOURS-1],battery_nodes[0])].varValue]+
#         [flow_vars[(battery_nodes[i],battery_nodes[i+1])].varValue for i in range(23)], label='bat usage')
ax3.title.set_text('Battery capacity = ' + str(cap_bat.varValue) + ' kW')

ax4.plot(sub_interval, [lost_charging[i] for i in interval_indices], label='E lost charging')
ax4.plot(sub_interval, [lost_discharging[i] for i in interval_indices], label='E lost discharging')

leg = ax1.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax2.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax3.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax4.legend(prop={'size': fontsize * 0.9}, loc='upper right')
# plt.show()

if create_log_file:
    path = os.getcwd()  # creates folder in logs and metadata
    now_ = datetime.now()
    dt_string_ = now_.strftime("%Y_%m_%d_%Hh%M_%S")
    path = path + '/logs_and_metadata/' + dt_string_  # new folder
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    if create_csv:
        with open(path + '/data.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data_dict.keys())
            writer.writerows(zip(*data_dict.values()))
        # with open(path+'/data.csv', 'w', newline='') as myfile:
        #    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #    wr.writerow(mylist)
    if create_tendance:
        with open(path + '/tendance.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(courbe_tendance_dict.keys())
            writer.writerows(zip(*courbe_tendance_dict.values()))
    if create_graphml_obj == 1:
        name = 'graph_' + str(start_hour) + '_' + str(end_hour - 1) + '.graphml'
        nx.write_graphml(G, path + '/' + name)
    if create_plot:
        fig.savefig(path + '/plot.png')
    if LpStatus[
        prob.status] == 'Optimal' and create_lp_file_if_feasible_and_less_than_49_hours and NUMBER_OF_HOURS < 49:
        print('creating LP file')
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%Hh%M")
        prob.writeLP(path + '/stanislawLP.lp')
    file = open(path + '/meta_data' + str(start_hour) + '_' + str(end_hour - 1) + '.txt', "w")
    file.write('Information concerning this particular LP\n')
    string_to_write = 'hours considered: [' + str(start_hour) + ',' + str(end_hour - 1) + '], including ' + str(
        end_hour - 1) + '\n'
    file.write(string_to_write)
    string_tmp = 'Constants:\n'
    for key in constants:
        string_tmp += key + ' = ' + str(constants[key][0]) + ' ' + str(constants[key][1]) + '\n'
    string_tmp += 'LpStatus: ' + LpStatus[prob.status] + '\n'
    string_tmp += 'optimized rated power of installation = ' + str(round(prated_solar.varValue, 3)) + ' kWp\n'
    string_tmp += 'optimized battery capacity: ' + str(round(cap_bat.varValue, 3)) + ' kWh\n'
    string_tmp += 'total cost: ' + str(round(value(prob.objective), 3)) + ' CHF\n'
    string_tmp += 'total cost if we only bought from the grid: ' + str(round(sum(
        [C_ENERGY_GRID[hour] * PCONS[hour] for hour in range(len(hours_considered))]) +
                                                                             P_CONS_MAX * C_POWER_GRID, 4)) + ' CHF\n'
    string_tmp += 'max power taken from the grid: ' + str(
        round(max_grid_cons.varValue, 3)) + ' kW (at a cost of ' + str(
        C_POWER_GRID) + ' CHF/kW)\n'
    string_tmp += 'max power taken from grid without microgrid (max consumption):' + str(round(P_CONS_MAX, 3)) + 'kW\n'
    string_tmp += 'time to solve LP = ' + str(round(time_to_solve, 3)) + ' seconds\n'
    string_tmp += 'time resolution [hours] = ' + str(resolution) + '\n'
    file.write(string_tmp)
    string_tmp = '\ncontinuous pv rated power optimization with'
    if optimize_with_gurobi:
        string_tmp += ' Gurobi'
    else:
        string_tmp += ' pulp free solver'
    file.write(string_tmp)
    file.close()

if circular_time:
    print(flow_vars[(battery_nodes[NUMBER_OF_HOURS-1],battery_nodes[0])].varValue)