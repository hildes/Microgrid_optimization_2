import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import xlrd
from datetime import date
from datetime import datetime
import gurobipy
import os
import csv

create_graphml_obj = 0  # do you want to create a graph object?
create_lp_file_if_feasible_and_less_than_49_hours = 0
print_variable_values_bool = 0
create_log_file = 1
create_csv = 1
create_plot = 1
optimize_with_gurobi = 1  # CBC is the default solver used by pulp


def import_planair_data():
    data_cons = pd.read_excel(
        r'/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization/Time_graph2/energy_data/Conso.xlsx')
    data_pv = pd.read_excel(
        r'/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization/Time_graph2/energy_data/PV.xlsx')
    return data_cons['kW'].values, data_pv['[kW]'].values


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
    elif node.startswith('ev_cons'):
        return int(node[7:])
    elif node.startswith('bat_ev'):
        return int(node[6:])
    elif node.startswith('ev_bat'):
        return int(node[6:])
    elif node.startswith('EV'):
        return int(node[2:])
    return -12


consumption_data, pv_data = import_planair_data()
# value, unit
# in comment typical value from Christian
constants = {'CAP_MIN_BAT': [0.0, 'kWh'],  # 0
             'CAP_MAX_BAT': [10, 'kWh'],  # 10
             'CAPEX_VARIABLE_BAT': [470, 'CHF/kWh'],  # 470
             'CAPEX_FIXED_BAT': [3700, 'CHF'],  # 3700
             'OPEX_VARIABLE_BAT': [0, 'CHF/year/kWh'],  # 0
             'OPEX_FIXED_BAT': [0, 'CHF/year'],  # 0
             'P_RATED_MIN_SOLAR': [0, 'kWp'],  # 0
             'P_RATED_MAX_SOLAR': [25, 'kWp'],  # 25
             'CAPEX_VARIABLE_SOLAR': [1200.0, 'CHF/kWp'],  # 1200.0  # CHF/kWp
             'CAPEX_FIXED_SOLAR': [10000.0, 'CHF'],  # 10000.0  # fixed investment costs in CHF
             'OPEX_VARIABLE_SOLAR': [25, 'CHF/year/kWh'],  # 25
             'OPEX_FIXED_SOLAR': [0, 'CHF/year'],  # 0
             'BUYING_PRICE_GRID': [0.2, 'CHF/kWh'],  # 0.2
             'SELLING_PRICE_GRID': [0.05, 'CHF/kWh'],  # 0.05
             'C_POWER_GRID': [80, 'CHF/kW'],  # 80
             'P_MAX_INJECTED_GRID': [10, 'kW'],  # 10
             'P_MAX_EXTRACTED_GRID': [10, 'kW'],  # 10
             'C_DISCHARGE_MAX_BAT': [1, 'kW/kWh'],  # 1
             'C_CHARGE_MAX_BAT': [1, 'kW/kWh'],  # 1
             'ETA_DISCHARGE_BAT': [0.95, ' '],  # 0.95
             'ETA_CHARGE_BAT': [0.95, ' '],  # 0.95
             'LIFETIME_BAT': [10, 'years'],  # 10
             'LIFETIME_SOLAR': [25, 'years'],  # 25
             'CAP_BAT_EV': [50, 'kWh'],
             'ETA_CHARGE_EV': [0.95, ' '],
             'ETA_DISCHARGE_EV': [0.95, ' '],
             'MIN_LEAVING_CHARGE_PERCENT_EV': [0.9, '%'],
             'REENTRY_CHARGE_PERCENT_EV': [0.25, '%'],
             'CCHARGE_MAX_EV': [1, 'kW/kWh'],
             'CDISCHARGE_MAX_EV': [1, 'kW/kWh']
             }
CAP_MIN_BAT = constants['CAP_MIN_BAT'][0]
CAP_MAX_BAT = constants['CAP_MAX_BAT'][0]
CAPEX_VARIABLE_BAT = constants['CAPEX_VARIABLE_BAT'][0]
CAPEX_FIXED_BAT = constants['CAPEX_FIXED_BAT'][0]
OPEX_VARIABLE_BAT = constants['OPEX_VARIABLE_BAT'][0]
OPEX_FIXED_BAT = constants['OPEX_FIXED_BAT'][0]
start_hour = 0
end_hour = 1*48
hours_considered = range(start_hour, end_hour)
hours_considered_indices = range(len(hours_considered))
PCONS = [consumption_data[i] for i in hours_considered]
NUMBER_OF_HOURS = end_hour - start_hour  # len(hours_considered)
P_CONS_MAX = max(PCONS)
# production curve for installation of 1kW rated power
PNORMSOLAR = [pv_data[i] for i in hours_considered]
P_RATED_MIN_SOLAR = constants['P_RATED_MIN_SOLAR'][0]
P_RATED_MAX_SOLAR = constants['P_RATED_MAX_SOLAR'][0]
PGENSOLAR = np.array([PNORMSOLAR[i] for i in range(len(PNORMSOLAR))])

CAPEX_VARIABLE_SOLAR = constants['CAPEX_VARIABLE_SOLAR'][0]
CAPEX_FIXED_SOLAR = constants['CAPEX_FIXED_SOLAR'][0]
OPEX_VARIABLE_SOLAR = constants['OPEX_VARIABLE_SOLAR'][0]
OPEX_FIXED_SOLAR = constants['OPEX_FIXED_SOLAR'][0]
C_POWER_GRID = constants['C_POWER_GRID'][0]
C_ENERGY_GRID = np.full(NUMBER_OF_HOURS, constants['BUYING_PRICE_GRID'][0])
CINJECTIONGRID = np.full(NUMBER_OF_HOURS, constants['SELLING_PRICE_GRID'][0])
P_MAX_INJECTED_GRID = constants['P_MAX_INJECTED_GRID'][0]
P_MAX_EXTRACTED_GRID = constants['P_MAX_EXTRACTED_GRID'][0]
C_DISCHARGE_MAX_BAT = constants['C_DISCHARGE_MAX_BAT'][0]
C_CHARGE_MAX_BAT = constants['C_CHARGE_MAX_BAT'][0]
ETA_DISCHARGE_BAT = constants['ETA_DISCHARGE_BAT'][0]
ETA_CHARGE_BAT = constants['ETA_CHARGE_BAT'][0]

LIFETIME_BAT = constants['LIFETIME_BAT'][0]
LIFETIME_SOLAR = constants['LIFETIME_SOLAR'][0]

CAP_BAT_EV = constants['CAP_BAT_EV'][0]
ETA_CHARGE_EV = constants['ETA_CHARGE_EV'][0]
ETA_DISCHARGE_EV = constants['ETA_DISCHARGE_EV'][0]
MIN_LEAVING_CHARGE_PERCENT_EV = constants['MIN_LEAVING_CHARGE_PERCENT_EV'][0]
REENTRY_CHARGE_PERCENT_EV = constants['REENTRY_CHARGE_PERCENT_EV'][0]
CCHARGE_MAX_EV = constants['CCHARGE_MAX_EV'][0]
CDISCHARGE_MAX_EV = constants['CDISCHARGE_MAX_EV'][0]

WEEK_DAY_PRESENCE = np.concatenate([np.full(7, 1), np.full(12, 0), np.full(5, 1)])
WEEKEND_DAY_PRESENCE = np.concatenate([np.full(9, 1), np.full(3, 0), np.full(2, 1), np.full(6, 0), np.full(4, 1)])
MINIMAL_CHARGE_WEEK_DAY_PERCENTAGE = np.concatenate([np.array([10]),np.full(5, 15),np.array([70]), np.full(12, 0), np.full(5, 15)])
MINIMAL_CHARGE_WEEKEND_DAY_PERCENTAGE = np.concatenate(
    [np.full(9, 15), np.full(3, 0), np.array([50,60]), np.full(6, 0), np.full(4, 15)])

HOURLY_PRESENCE_WEEK = np.concatenate([np.tile(WEEK_DAY_PRESENCE, 5), np.tile(WEEKEND_DAY_PRESENCE, 2)])
HOURLY_MINIMAL_CHARGE_WEEK_PERCENTAGE = np.concatenate(
    [np.tile(MINIMAL_CHARGE_WEEK_DAY_PERCENTAGE, 5), np.tile(MINIMAL_CHARGE_WEEKEND_DAY_PERCENTAGE, 2)])
NUMBER_OF_WEEKS = int(NUMBER_OF_HOURS/(24*7))
REMAINING_DAYS = int((NUMBER_OF_HOURS - NUMBER_OF_WEEKS*24*7)/24)
REMAINING_HOURS = int(NUMBER_OF_HOURS-NUMBER_OF_WEEKS*(24*7)-REMAINING_DAYS*(24))
number_of_days = int(NUMBER_OF_HOURS/24)
last_hours_presence = np.array([])
last_hours_minimal_charge_perc = []
if number_of_days % 7 in [5,6]:
    last_hours_presence = np.array([WEEKEND_DAY_PRESENCE[i] for i in range(REMAINING_HOURS)])
    last_hours_minimal_charge_perc = np.array([MINIMAL_CHARGE_WEEKEND_DAY_PERCENTAGE[i] for i in range(REMAINING_HOURS)])
else:
    last_hours_presence = np.array([WEEK_DAY_PRESENCE[i] for i in range(REMAINING_HOURS)])
    last_hours_minimal_charge_perc = np.array([MINIMAL_CHARGE_WEEK_DAY_PERCENTAGE[i] for i in range(REMAINING_HOURS)])

HOURLY_PRESENCE_YEAR = np.concatenate([np.tile(HOURLY_PRESENCE_WEEK, NUMBER_OF_WEEKS),
                                       np.tile(WEEK_DAY_PRESENCE, REMAINING_DAYS), last_hours_presence])
HOURLY_MINIMAL_CHARGE_YEAR = np.concatenate(
    [np.tile(HOURLY_MINIMAL_CHARGE_WEEK_PERCENTAGE, NUMBER_OF_WEEKS),
     np.tile(MINIMAL_CHARGE_WEEK_DAY_PERCENTAGE, REMAINING_DAYS),
     last_hours_minimal_charge_perc]) * CAP_BAT_EV / 100 # a year has
# exactly (not really) 52 weeks and 1 day (365 - 7*52 = 1)

HOURLY_PRESENCE_YEAR_HOURS = []
for i in range(end_hour):
    if HOURLY_PRESENCE_YEAR[i] == 1:
        HOURLY_PRESENCE_YEAR_HOURS += [i]
LEAVING_THE_GARAGE_HOURS = []
ENTERING_THE_GARAGE_HOURS = []
for i in range(end_hour - 1):
    if HOURLY_PRESENCE_YEAR[i] == 1 and HOURLY_PRESENCE_YEAR[i + 1] == 0:
        LEAVING_THE_GARAGE_HOURS += [i]
    if HOURLY_PRESENCE_YEAR[i] == 0 and HOURLY_PRESENCE_YEAR[i + 1] == 1:
        ENTERING_THE_GARAGE_HOURS += [i+1]

start_time_graph = time.time()  # start counting

G = nx.DiGraph()
# --------------adding nodes--------------
G.add_nodes_from(['supersource', 'supersupersourcePV', 'supersourcePV', 'supersink', 'EV_sink', 'EV_source'])
pv_nodes = {hour: 'PV' + str(hour) for hour in hours_considered}
battery_nodes = {hour: 'Battery' + str(hour) for hour in hours_considered}
consumption_nodes = {hour: 'Consumption' + str(hour) for hour in hours_considered}
pv_bat_nodes = {hour: 'pv_bat' + str(hour) for hour in hours_considered}
bat_cons_nodes = {hour: 'bat_cons' + str(hour) for hour in hours_considered}
ev_nodes = {}
ev_cons_nodes = {}
bat_ev_nodes = {}
ev_bat_nodes = {}
for hour in HOURLY_PRESENCE_YEAR_HOURS:
    ev_nodes[hour] = 'EV' + str(hour)
    ev_cons_nodes[hour] = 'ev_cons' + str(hour)
    bat_ev_nodes[hour] = 'bat_ev' + str(hour)
    ev_bat_nodes[hour] = 'ev_bat' + str(hour)
G.add_nodes_from(list(pv_nodes.values()) + list(battery_nodes.values()) + list(consumption_nodes.values()) +
                 list(pv_bat_nodes.values()) + list(bat_cons_nodes.values()) + list(ev_nodes.values()) +
                 list(ev_cons_nodes.values()) + list(bat_ev_nodes.values()) + list(ev_bat_nodes.values()))
# --------------adding arcs---------------

supersource_cons_edges = [('supersource', consumption_nodes[key]) for key in consumption_nodes]
supsrcPV_PV_edges = [('supersourcePV', pv_nodes[key]) for key in pv_nodes]
supsupsrcPV_supsrcPV_edge = ('supersupersourcePV', 'supersourcePV')
pv_cons_edges = [(pv_nodes[hour], consumption_nodes[hour]) for hour in hours_considered_indices]
pv_pv_bat_edges = [(pv_nodes[hour], pv_bat_nodes[hour]) for hour in hours_considered_indices]
pv_bat_bat_edges = [(pv_bat_nodes[hour], battery_nodes[hour]) for hour in hours_considered_indices]
print(pv_bat_bat_edges)
bat_bat_cons_edges = [(battery_nodes[hour], bat_cons_nodes[hour]) for hour in hours_considered_indices]
bat_cons_cons_edges = [(bat_cons_nodes[hour], consumption_nodes[hour]) for hour in hours_considered_indices]
bat_bat_edges = [(battery_nodes[i], battery_nodes[i + 1]) for i in range(NUMBER_OF_HOURS)[:-1]]
cons_supersink_edges = [(consumption_nodes[hour], 'supersink') for hour in range(NUMBER_OF_HOURS)]
pv_supersink_edges = [(pv_nodes[hour], 'supersink') for hour in range(NUMBER_OF_HOURS)]
ev_ev_edges = {}
grid_ev_edges = {}
ev_ev_bat_edges = {}
for i in hours_considered:
    if i in ev_nodes and i+1 in ev_nodes:
        ev_ev_edges[i] = (ev_nodes[i],ev_nodes[i+1])
    if i in ev_nodes:
        grid_ev_edges[i] = ('supersource', ev_nodes[i])
        ev_ev_bat_edges[i] = (ev_nodes[i], ev_bat_nodes[i])
ev_ev_cons_edges = [(ev_nodes[key], ev_cons_nodes[key]) for key in ev_nodes]
ev_cons_cons_edges = [(ev_cons_nodes[key], consumption_nodes[key]) for key in ev_nodes]
pv_ev_edges = [(pv_nodes[key], ev_nodes[key]) for key in ev_nodes]
ev_sink_edges = [(ev_nodes[hour], 'EV_sink') for hour in LEAVING_THE_GARAGE_HOURS]
source_ev_edges = [('EV_source', ev_nodes[hour]) for hour in ENTERING_THE_GARAGE_HOURS]
bat_bat_ev_edges = [(battery_nodes[key], bat_ev_nodes[key]) for key in ev_nodes]
bat_ev_ev_edges = [(bat_ev_nodes[key], ev_nodes[key]) for key in ev_nodes]
ev_bat_bat_edges = [(ev_bat_nodes[key], battery_nodes[key]) for key in ev_nodes]

G.add_edges_from([supsupsrcPV_supsrcPV_edge] + supersource_cons_edges + supsrcPV_PV_edges
                 + pv_cons_edges + pv_pv_bat_edges + pv_bat_bat_edges + bat_bat_cons_edges + bat_cons_cons_edges +
                 bat_bat_edges + cons_supersink_edges + pv_supersink_edges + list(ev_ev_edges.values()) +
                 ev_ev_cons_edges + ev_cons_cons_edges + list(grid_ev_edges.values()) + pv_ev_edges + ev_sink_edges + source_ev_edges+
                 bat_bat_ev_edges + bat_ev_ev_edges + list(ev_ev_bat_edges.values()) + ev_bat_bat_edges)
print('--- %s seconds --- to create graph' % (time.time() - start_time_graph))

fixed_flow_bounds = {}
fixed_arc_flow_cost = {}

for e in supersource_cons_edges:
    fixed_arc_flow_cost[e] = C_ENERGY_GRID[node_hour(e[1])]  # 'supersource','Consumption' + str(hour)
for e in cons_supersink_edges:
    fixed_flow_bounds[e] = [PCONS[node_hour(e[0]) - start_hour], 5*P_CONS_MAX + 999]  # upper bound could just be infinity
for e in pv_supersink_edges:
    fixed_arc_flow_cost[e] = -CINJECTIONGRID[node_hour(e[0]) - start_hour]
    fixed_flow_bounds[e] = [0, P_MAX_INJECTED_GRID]

for key in grid_ev_edges:
    fixed_arc_flow_cost[grid_ev_edges[key]] = C_ENERGY_GRID[key]

start_time_constraints = time.time()  # start counting

flow_vars = LpVariable.dicts('flow', G.edges(), 0, None, cat='Continuous')
prated_solar = LpVariable('rated power', P_RATED_MIN_SOLAR, P_RATED_MAX_SOLAR, cat='Continuous')
cap_bat = LpVariable('cap_bat', CAP_MIN_BAT, CAP_MAX_BAT, cat='Continuous')
non_zero_pv = LpVariable('non_zero_pv', 0, 1, cat='Binary')
non_zero_bat = LpVariable('non_zero_bat', 0, 1, cat='Binary')
max_grid_cons = LpVariable('max_grid_cons', cat='Continuous')

prob = LpProblem('Energy flow problem', LpMinimize)
# Creates the objective function
prob += lpSum([flow_vars[arc] * fixed_arc_flow_cost[arc] for arc in fixed_arc_flow_cost] +
              [non_zero_pv * ((CAPEX_FIXED_SOLAR) / (LIFETIME_SOLAR) + OPEX_FIXED_SOLAR) + prated_solar * (
                      CAPEX_VARIABLE_SOLAR / LIFETIME_SOLAR + OPEX_VARIABLE_SOLAR)] +
              [non_zero_bat * (
                      (CAPEX_FIXED_BAT * len(hours_considered)) / (LIFETIME_BAT * 8760) + OPEX_FIXED_BAT) + cap_bat * (
                       CAPEX_VARIABLE_BAT * len(hours_considered) / (LIFETIME_BAT * 8760) + OPEX_VARIABLE_BAT)] +
              [max_grid_cons * C_POWER_GRID])

# fixed investment costs if there is >0 kWp installed
(fixed_mins, fixed_maxs) = splitDict(fixed_flow_bounds)
for arc_key in fixed_flow_bounds:
    # prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintLE,rhs = fixed_maxs[arc_key])
    # prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintGE,rhs = fixed_mins[arc_key])
    # prob += flow_vars[arc_key] <= fixed_maxs[arc_key]
    # prob += fixed_mins[arc_key] <= flow_vars[arc_key]
    flow_vars[arc_key].bounds(fixed_mins[arc_key], fixed_maxs[arc_key])

for h in hours_considered:
    if h in ev_nodes:
        prob += lpSum([flow_vars[supersource_cons_edges[h]], flow_vars[grid_ev_edges[h]]]) <= P_MAX_EXTRACTED_GRID # todo: new constraint check that it's correct
    else:
        prob += flow_vars[supersource_cons_edges[h]] <= P_MAX_EXTRACTED_GRID
for e in bat_bat_edges:
    prob += flow_vars[e] <= cap_bat, 'battery capacity' + str(e[0])


for h in hours_considered:
    if h in ev_nodes:
        prob += lpSum([flow_vars[pv_pv_bat_edges[h]], flow_vars[ev_ev_bat_edges[h]]]) <= C_CHARGE_MAX_BAT * cap_bat, 'BAT charging rate' + str(h)
    else:
        prob += flow_vars[pv_pv_bat_edges[h]] <= C_CHARGE_MAX_BAT * cap_bat, 'BAT charging rate' + str(h)
#for e in pv_pv_bat_edges:
#    prob += flow_vars[e] <= C_CHARGE_MAX_BAT * cap_bat, 'BAT charging rate' + str(e[0])#todo: battery can be charged from PV or EV or both!


for e in bat_bat_cons_edges:
    prob += flow_vars[e] <= C_DISCHARGE_MAX_BAT * cap_bat, 'BAT discharging rate' + str(e[0])
for e in supersource_cons_edges:  # max power cost
    prob += flow_vars[e] <= max_grid_cons
for key in ev_ev_edges:
    prob += flow_vars[ev_ev_edges[key]] <= CAP_BAT_EV
    prob += flow_vars[ev_ev_edges[key]] >= HOURLY_MINIMAL_CHARGE_YEAR[key]  # todo: make sure this is correct
for e in ev_ev_cons_edges:
    prob += flow_vars[e] <= CDISCHARGE_MAX_EV * CAP_BAT_EV, 'EV discharging rate' + str(node_hour(e[0]))
for key in grid_ev_edges:
    prob += flow_vars[grid_ev_edges[key]] <= CCHARGE_MAX_EV * CAP_BAT_EV, 'EV charging rate' + str(key)
for e in pv_ev_edges:
    prob += flow_vars[e] <= CCHARGE_MAX_EV * CAP_BAT_EV
for e in ev_sink_edges:
    prob += flow_vars[e] >= MIN_LEAVING_CHARGE_PERCENT_EV * CAP_BAT_EV, 'leaving garage min charge' + str(
        node_hour(e[0]))
for e in source_ev_edges:
    prob += flow_vars[e] == REENTRY_CHARGE_PERCENT_EV * CAP_BAT_EV, 'reentry to garage max charge' + str(
        node_hour(e[1]))

total_sun_gen = sum(PNORMSOLAR)
prob += flow_vars[supsupsrcPV_supsrcPV_edge] <= total_sun_gen * prated_solar
prob += prated_solar <= non_zero_pv * P_RATED_MAX_SOLAR
prob += cap_bat <= non_zero_bat * CAP_MAX_BAT
for e in supsrcPV_PV_edges:
    prob += flow_vars[e] <= PNORMSOLAR[node_hour(e[1])] * prated_solar
# flow conservation
for n in G.nodes():
    if not (n.startswith('super')
            or n.startswith('pv_bat')
            or n.startswith('bat_cons')
            or n.startswith('supersourcePV') or n.startswith('ev_bat') or n.startswith('bat_ev') or
            n.startswith('EV_sink') or n.startswith('EV_source') or n.startswith('ev_cons')):
        prob += lpSum(flow_vars[ingoing] for ingoing in G.in_edges(n)) == \
                lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
# loss of flow because of the battery
for n in pv_bat_nodes.values():
    prob += lpSum(flow_vars[ingoing] * ETA_CHARGE_BAT for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
for n in bat_cons_nodes.values():
    prob += lpSum(flow_vars[ingoing] * ETA_DISCHARGE_BAT for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
for n in ev_bat_nodes.values():
    prob += lpSum(flow_vars[ingoing] * ETA_DISCHARGE_EV * ETA_CHARGE_BAT for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
for n in bat_ev_nodes.values():
    prob += lpSum(flow_vars[ingoing] * ETA_DISCHARGE_BAT * ETA_CHARGE_EV for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))
for n in ev_cons_nodes.values():
    prob += lpSum(flow_vars[ingoing] * ETA_DISCHARGE_EV for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))

print('--- %s seconds --- to add constraints' %
      (time.time() - start_time_constraints))

start_time = time.time()  # start counting
if optimize_with_gurobi:
    prob.solve(GUROBI())
else:
    prob.solve()

time_to_solve = time.time() - start_time
print('--- %s seconds --- to solve' % time_to_solve)

if LpStatus[prob.status] == 'Optimal' and create_lp_file_if_feasible_and_less_than_49_hours and NUMBER_OF_HOURS < 49:
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hh%M")
    prob.writeLP('lp_files/' + dt_string + '_LP.lp')

print('total Cost of microgrid = ', value(prob.objective))
''' # hard to calculate in one folrmula as the EV battery can play a complex role in optimizing usage of grid and EV
print('total theoretical cost without microgrid: '
      + str(sum([C_ENERGY_GRID[hour] * PCONS[hour] for hour in hours_considered_indices] +
                [C_ENERGY_CITY[h] * REENTRY_CHARGE_PERCENT_EV * CAP_BAT_EV for h in ENTERING_THE_GARAGE_HOURS]) +
            P_CONS_MAX * C_POWER_GRID) + ' CHF')
'''
if print_variable_values_bool == 1:
    print_variable_values(prob)
print('optimized battery capacity: ', cap_bat.varValue, ' kWh')
print('optimized rated power of installation = ', prated_solar.varValue, ' kWp')

'''# if infeasible, this points to an impossible constraint
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

sub_interval = range(start_hour, start_hour+48)
hours_considered_set = set(hours_considered)
interval = list(hours_considered_set.intersection(sub_interval))
interval_indices = range(len(interval))

# interval = range(len(hours_considered))
interval = sub_interval
interval_indices = sub_interval

data_dict = {}

grid_cons = [flow_vars[supersource_cons_edge].varValue for supersource_cons_edge in supersource_cons_edges]
data_dict['grid_cons'] = grid_cons
pv_grid = [flow_vars[pv_supersink_edge].varValue for pv_supersink_edge in pv_supersink_edges]
data_dict['pv_grid'] = pv_grid
battery_cons = [flow_vars[bat_cons_cons_edge].varValue for bat_cons_cons_edge in bat_cons_cons_edges]
data_dict['battery_cons'] = battery_cons
lost_discharging = [flow_vars[bat_bat_cons_edge].varValue * (1 - ETA_DISCHARGE_BAT) for bat_bat_cons_edge in
                    bat_cons_cons_edges]
data_dict['lost_discharging'] = lost_discharging
pv_battery = [flow_vars[e].varValue for e in pv_pv_bat_edges]
data_dict['pv_battery'] = pv_battery


lost_charging = [flow_vars[e].varValue * (1 - ETA_CHARGE_BAT) for e in pv_pv_bat_edges]
data_dict['lost_charging'] = lost_charging
pv_cons = [flow_vars[pv_cons_edge].varValue for pv_cons_edge in pv_cons_edges]
data_dict['pv_cons'] = pv_cons
battery_usage = [0] + [flow_vars[bat_bat_edge].varValue for bat_bat_edge in bat_bat_edges]
data_dict['battery_usage'] = battery_usage
data_dict['consumption_data'] = consumption_data
data_dict['normalized_pv'] = pv_data
data_dict['EV_presence'] = HOURLY_PRESENCE_YEAR
ev_ev = np.full(NUMBER_OF_HOURS, 0)
for key in ev_ev_edges:
    ev_ev[key] = flow_vars[ev_ev_edges[key]].varValue
data_dict['ev_ev'] = ev_ev
ev_cons = np.full(NUMBER_OF_HOURS, 0)
# ev_cons_cons_edges = [(ev_cons_nodes[key], consumption_nodes[key]) for key in ev_nodes]
for h in ev_nodes:
    edge = (ev_cons_nodes[h], consumption_nodes[h])
    ev_cons[h] = flow_vars[edge].varValue
data_dict['ev_cons'] = ev_cons

ev_battery = np.full(NUMBER_OF_HOURS,0)
for key in ev_nodes:
    ev_battery[key] = flow_vars[(ev_nodes[key], ev_bat_nodes[key])].varValue
data_dict['ev_battery'] = ev_battery

for h in hours_considered:
    if pv_battery[h] > 0 and battery_cons[h] > 0:
        print(h, ' charging and discharging at the same time')
        break
else:
    print('never charging and discharging at the same time')
print('max_grid_cons * C_POWER_GRID = ', max_grid_cons.varValue, '*', C_POWER_GRID, '=',
      max_grid_cons.varValue * C_POWER_GRID)

pltsize = 0.48
pltratio = 0.35
fontsize = 12
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(
    len(interval) * pltsize, len(interval) * pltsize * pltratio))
ax1.title.set_text(
    'Rated power of PV installation = ' + str(prated_solar) + ' kWp, total cost: ' + str(value(prob.objective))[
                                                                                     0:12] + ' CHF')

ax1.set_ylabel('Solar ', fontsize=fontsize)
ax1.plot(interval, [pv_battery[i] for i in interval_indices], label='PV -> battery')
ax1.plot(interval, [pv_grid[i]
                    for i in interval_indices], label='PV -> grid', marker='+')
ax1.plot(interval, [pv_cons[i]
                    for i in interval_indices], label='PV -> cons', marker='x')

ax1.plot(interval, [PGENSOLAR[i] for i in interval_indices], label='normalized solar curve', color='y')
sum_PV_into_ = [pv_battery[i] + pv_grid[i] + pv_cons[i]
                for i in range(NUMBER_OF_HOURS)]
ax1.plot(interval, [sum_PV_into_[i] for i in interval_indices], label='sum PV ->')

ax2.plot(interval, [battery_cons[i]
                    for i in interval_indices], label='battery -> cons')
ax2.plot(interval, [pv_cons[i]
                    for i in interval_indices], label='PV -> cons', marker='x')
ax2.set_ylabel('consumption', fontsize=fontsize)
ax2.plot(interval, [PCONS[i]
                    for i in interval_indices], label='consumption', marker='x')
ax2.plot(interval, [grid_cons[i]
                    for i in interval_indices], label='grid -> cons', marker='+')
sum_into_cons = [pv_cons[i] + grid_cons[i] + battery_cons[i] + ev_cons[i]
                 for i in range(len(grid_cons))]
ax2.plot(interval, [sum_into_cons[i] for i in interval_indices], label='into cons')
ax2.title.set_text('Energy cost: ' + str(C_ENERGY_GRID[0]) +
                   ', can be sold for: ' + str(CINJECTIONGRID[0]))

ax3.plot(interval, [-battery_cons[i]
                    for i in interval_indices], label='bat -> cons', color='r')
ax3.plot(interval, [pv_battery[i]
                    for i in interval_indices], label='PV -> bat', color='lawngreen')
ax3.plot(interval, [ev_battery[i]
                    for i in interval_indices], label='EV -> bat', color='seagreen')
ax3.plot(interval, np.full(len(interval_indices), cap_bat.varValue), label='bat capacity',color='darkorange')
ax3.set_ylabel('battery usage', fontsize=fontsize)
ax3.set_xlabel('hours', fontsize=fontsize)
ax3.plot(interval, [battery_usage[i] for i in interval_indices], label='bat usage')
ax3.title.set_text('Battery capacity = ' + str(cap_bat.varValue) + ' kW')

ax4.plot(interval, [lost_charging[i] for i in interval_indices], label='E lost charging')
ax4.plot(interval, [lost_discharging[i] for i in interval_indices], label='E lost discharging',color='r')

ax5.plot(interval, np.full(len(interval_indices), CAP_BAT_EV), label='ev bat capacity')
ax5.plot(interval, [HOURLY_PRESENCE_YEAR[h] for h in interval_indices], label='car presence')
ax5.plot(interval, [HOURLY_MINIMAL_CHARGE_YEAR[h] for h in interval_indices], label='minimal charge')
ax5.plot(interval, [ev_ev[h] for h in interval_indices], label='ev->ev')
ax5.plot(interval, [ev_cons[h] for h in interval_indices], label = 'ev->cons')

leg = ax1.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax2.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax3.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax4.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax5.legend(prop={'size': fontsize * 0.9}, loc='upper right')
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
    string_tmp += '\nLpStatus: ' + LpStatus[prob.status] + '\n'
    string_tmp += 'optimized rated power of installation = ' + str(prated_solar.varValue) + ' kWp\n'
    string_tmp += 'optimized battery capacity: ' + str(cap_bat.varValue) + ' kWh \n'
    string_tmp += 'total cost: ' + str(value(prob.objective))[0:12] + ' CHF\n'
    string_tmp += 'total cost if we only bought from the grid: ' + \
                  str(sum([(C_ENERGY_GRID[hour] + HOURLY_MINIMAL_CHARGE_YEAR[hour]) * PCONS[hour] for hour in
                           hours_considered_indices]) + P_CONS_MAX * C_POWER_GRID) + ' CHF\n'
    string_tmp += 'max power taken from the grid: ' + str(max_grid_cons.varValue) + ' kW (at a cost of ' + str(
        C_POWER_GRID) + ' CHF/kW)\n'
    string_tmp += 'max power taken from grid without microgrid (max consumption): ' + str(P_CONS_MAX) + 'kW\n'
    string_tmp += 'time to solve LP = ' + str(time_to_solve) + ' seconds \n'
    file.write(string_tmp)
    string_tmp = '\ncontinuous pv rated power optimization with'
    if optimize_with_gurobi:
        string_tmp += ' Gurobi'
    else:
        string_tmp += ' pulp free solver'
    file.write(string_tmp)
    file.close()
