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
constants = {'CAPMINBAT': [0.0, 'kWh'],#0
             'CAPMAXBAT': [11.5, 'kWh'],#10
             'CAPEXVARIABLEBAT': [29, 'CHF/kWh'],#470
             'CAPEXFIXEDBAT': [3.7, 'CHF'],#3700
             'PRATEDMINSOLAR': [0, 'kWp'],#0
             'PRATEDMAXSOLAR': [25, 'kWp'],#25
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
G.add_nodes_from(['supersource', 'supersupersourcePV', 'supersink'])
supsrcPV_nodes = ['supersourcePV' + str(i) for i in range(PRATEDMAXSOLAR)]
pv_nodes = ['PV' + str(hour) for hour in hours_considered]
battery_nodes = ['Battery' + str(hour) for hour in hours_considered]
consumption_nodes = ['Consumption' + str(hour) for hour in hours_considered]
G.add_nodes_from(supsrcPV_nodes + pv_nodes + battery_nodes + consumption_nodes)
# --------------adding arcs---------------
supsupsrcPV_supsrcPV_edges = [('supersupersourcePV', supsrcPV_node)
                              for supsrcPV_node in supsrcPV_nodes]
supersource_cons_edges = [('supersource', consumption_node)
                          for consumption_node in consumption_nodes]
supsrcPVx_PV_edges = []
for supsrcPV_node in supsrcPV_nodes:
    supsrcPVx_PV_edges += [(supsrcPV_node, pv_node) for pv_node in pv_nodes]
pv_cons_edges = [(pv_nodes[hour], consumption_nodes[hour])
                 for hour in range(NUMBER_OF_HOURS)]
pv_bat_edges = [(pv_nodes[hour], battery_nodes[hour])
                for hour in range(NUMBER_OF_HOURS)]
bat_cons_edges = [(battery_nodes[hour], consumption_nodes[hour])
                  for hour in range(NUMBER_OF_HOURS)]
bat_bat_edges = [(battery_nodes[i], battery_nodes[i + 1])
                 for i in range(NUMBER_OF_HOURS)[:-1]]
pv_supersink_edges = [(pv_nodes[hour], 'supersink')
                      for hour in range(NUMBER_OF_HOURS)]
G.add_edges_from(supsupsrcPV_supsrcPV_edges + supersource_cons_edges +
                 supsrcPVx_PV_edges + pv_cons_edges + pv_bat_edges + bat_cons_edges +
                 bat_bat_edges + pv_supersink_edges)
print('--- %s seconds --- to create graph' % (time.time() - start_time_graph))

fixed_flow_bounds = {}
fixed_arc_flow_cost = {}
# defining fixed bounds and costs
# supsupsrcPV_supsrcPV_edges prob += flow_vars[e] <= total_sun_gen * pv_activation_vars[e]
for e in supersource_cons_edges:
    fixed_flow_bounds[e] = [0, PMAXEXTRACTEDGRID]
    fixed_arc_flow_cost[e] = CENERGYGRID[node_hour(e[1])-start_hour] #'supersource','Consumption' + str(hour)

for e in supsrcPVx_PV_edges:
    fixed_flow_bounds[e] = [0,PGENSOLAR[node_hour(e[1])-start_hour]]
for e in pv_cons_edges:
    fixed_flow_bounds[e] = [0,PCONSMAX]
# pv_bat_edges prob += flow_vars[e] <= CCHARGEMAXBAT * CAPBAT
# bat_cons_edges prob += flow_vars[e] <= CDISCHARGEMAXBAT * CAPBAT
# bat_bat_edges prob += flow_vars[e] <= CAPBAT
for e in pv_supersink_edges:
    fixed_arc_flow_cost[e]=-CINJECTIONGRID[node_hour(e[0])-start_hour]
    fixed_flow_bounds[e]=[0, PMAXINJECTEDGRID]

start_time_constraints = time.time()  # start counting

# [(('supersupersourcePV', 'supersourcePV' + str(i))) for i in
# range(PRATEDMAXSOLAR)]  # should be replaced by range(PRATEDMINSOLAR,PRATEDMAXSOLAR)
# because the first PRATEDMINSOLAR edges are going to be activated by default
# for now PRATEDMINSOLAR is zero anyway


flow_vars = LpVariable.dicts('flow', G.edges(), 0, None, cat='Continuous')
pv_activation_vars = LpVariable.dicts(
    'activation', supsupsrcPV_supsrcPV_edges, cat='Binary')
CAPBAT = LpVariable('cap_bat', CAPMINBAT, CAPMAXBAT, cat='Continuous')
non_zero_pv = LpVariable('non_zero_pv',0,1,cat='Binary')
(fixed_mins, fixed_maxs) = splitDict(fixed_flow_bounds)

prob = LpProblem('Energy flow problem', LpMinimize)
# Creates the objective function
prob += lpSum([flow_vars[arc] * fixed_arc_flow_cost[arc] for arc in fixed_arc_flow_cost] +
              [pv_activation_vars[edge] * CAPEXVARIABLESOLAR for edge in supsupsrcPV_supsrcPV_edges] +
               [CAPBAT * CAPEXVARIABLEBAT] +
               [non_zero_pv * CAPEXFIXEDSOLAR])

# fixed investment costs if there is >0 kWp installed

for arc_key in fixed_flow_bounds:
    #prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintLE,rhs = fixed_maxs[arc_key])
    #prob += pulp.LpConstraint(flow_vars[arc_key], sense = pulp.LpConstraintGE,rhs = fixed_mins[arc_key])
    #prob += flow_vars[arc_key] <= fixed_maxs[arc_key]
    #prob += fixed_mins[arc_key] <= flow_vars[arc_key]
    flow_vars[arc_key].bounds(fixed_mins[arc_key],fixed_maxs[arc_key])
for e in bat_bat_edges:
    prob += flow_vars[e] <= CAPBAT, 'battery capacity'+str(e[0])
for e in pv_bat_edges:
    prob += flow_vars[e] <= CCHARGEMAXBAT * CAPBAT, 'charging rate'+str(e[0])
for e in bat_cons_edges:
    prob += flow_vars[e] <= CDISCHARGEMAXBAT * CAPBAT, 'discharging rate'+str(e[0])
for cons in consumption_nodes:
    prob += lpSum(flow_vars[into_cons] for into_cons in G.in_edges(cons)) >= PCONS[node_hour(cons)], 'energy need ' +str(node_hour(cons))+ ' '
total_sun_gen = sum(PGENSOLAR)
for e in supsupsrcPV_supsrcPV_edges:
    if e == supsupsrcPV_supsrcPV_edges[0]:
        prob += total_sun_gen * pv_activation_vars[e] <= flow_vars[e]
    prob += flow_vars[e] <= total_sun_gen * pv_activation_vars[e]
    prob += non_zero_pv >= pv_activation_vars[e]
# flow conservation
for n in G.nodes():
    if not (n.startswith('super') or n.startswith('Consumption')) or n.startswith('supersourcePV'):
        prob += lpSum(flow_vars[ingoing] for ingoing in G.in_edges(n)) == \
            lpSum(flow_vars[outgoing] for outgoing in G.out_edges(n))

print('--- %s seconds --- to add constraints' %
      (time.time() - start_time_constraints))


start_time = time.time()  # start counting
# prob.solve()
prob.solve(GUROBI())
time_to_solve = time.time() - start_time
print('--- %s seconds --- to solve' % time_to_solve)


print('Total Cost of Energy = ', value(prob.objective))
if print_variable_values_bool == 1:
    print_variable_values(prob)
print('optimized battery capacity: ', CAPBAT.varValue)

PRATEDSOLAR = 0
for e in supsupsrcPV_supsrcPV_edges:
    PRATEDSOLAR += pv_activation_vars[e].varValue
print('Rated power of installation = ', PRATEDSOLAR, ' kWp')



''' # if infeasible, this points to an impossible constraint
print('The model is infeasible; computing IIS')
prob.writeLP('infeasible_LP.lp')
m = gurobipy.read('/Users/stanislashildebrandt/Documents/GitHub/Microgrid_optimization/Time_graph2/infeasible_LP.lp')
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

sub_interval = range(48)
hours_considered_set = set(hours_considered)
interval = list(hours_considered_set.intersection(sub_interval))
#interval = range(len(hours_considered))

grid_cons = [flow_vars[supersource_cons_edge].varValue for supersource_cons_edge in supersource_cons_edges]
PV_grid = [flow_vars[pv_supersink_edge].varValue for pv_supersink_edge in pv_supersink_edges]
battery_cons = [flow_vars[bat_cons_edge].varValue for bat_cons_edge in bat_cons_edges]
PV_battery = [flow_vars[pv_bat_edge].varValue for pv_bat_edge in pv_bat_edges]
PV_cons = [flow_vars[pv_cons_edge].varValue for pv_cons_edge in pv_cons_edges]
battery_usage = [0] + [flow_vars[bat_bat_edge].varValue for bat_bat_edge in bat_bat_edges]

pltsize = 0.45
pltratio = 0.25
fontsize = 12
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(
    len(interval) * pltsize, len(interval) * pltsize * pltratio))
ax1.title.set_text(
    'Rated power of PV installation = ' + str(PRATEDSOLAR) + ' kWp, total cost: ' + str(value(prob.objective))[
        0:12] + ' CHF')

ax1.set_ylabel('Solar ', fontsize=fontsize)
ax1.plot(interval, [PV_battery[i] for i in interval], label='PV -> battery')
ax1.plot(interval, [PV_grid[i]
                    for i in interval], label='PV -> grid', marker='+')
ax1.plot(interval, [PV_cons[i]
                    for i in interval], label='PV -> cons', marker='x')

ax1.plot(interval, [PGENSOLAR[i] for i in interval], label='normalized solar curve', color='y')
sum_PV_into_ = [PV_battery[i] + PV_grid[i] + PV_cons[i]
                for i in range(NUMBER_OF_HOURS)]
ax1.plot(interval, [sum_PV_into_[i] for i in interval], label='sum PV ->')

ax2.plot(interval, [battery_cons[i]
                    for i in interval], label='battery -> cons')
ax2.plot(interval, [PV_cons[i]
                    for i in interval], label='PV -> cons', marker='x')
ax2.set_ylabel('consumption', fontsize=fontsize)
ax2.plot(interval, [PCONS[i]
                    for i in interval], label='consumption', marker='x')
ax2.plot(interval, [grid_cons[i]
                    for i in interval], label='grid -> cons', marker='+')
sum_into_cons = [PV_cons[i] + grid_cons[i] + battery_cons[i]
                 for i in range(len(grid_cons))]
ax2.plot(interval, [sum_into_cons[i] for i in interval], label='into cons')
ax2.title.set_text('Energy cost: ' + str(buying_price) +
                   ', can be sold for: ' + str(selling_price))

ax3.plot(interval, [-battery_cons[i]
                    for i in interval], label='bat -> cons', color='r')
ax3.plot(interval, [PV_battery[i]
                    for i in interval], label='PV -> bat', color='g')
ax3.plot(interval, np.full(len(interval), CAPBAT.varValue),label='bat capacity')
ax3.set_ylabel('battery usage', fontsize=fontsize)
ax3.set_xlabel('hours', fontsize=fontsize)
ax3.plot(interval, [battery_usage[i] for i in interval],label='bat usage')
ax3.title.set_text('Battery capacity = ' + str(CAPBAT.varValue) + ' kW')

leg = ax1.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax2.legend(prop={'size': fontsize * 0.9}, loc='upper right')
leg = ax3.legend(prop={'size': fontsize * 0.9}, loc='upper right')
#plt.show()

if create_log_file:
    path = os.getcwd() # creates folder in logs and metadata
    now_ = datetime.now()
    dt_string_ = now_.strftime("%d_%m_%Y_%Hh%M_%S")
    path = path+'/logs_and_metadata/'+dt_string_ # new folder
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    if create_graphml_obj == 1:
        name = 'graph_'+str(start_hour)+'_'+str(end_hour-1)+'.graphml'
        nx.write_graphml(G, path+'/'+name)
    fig.savefig(path+'/plot.png')
    if LpStatus[prob.status] == 'Optimal' and create_lp_file_if_feasible_and_less_than_49_hours and NUMBER_OF_HOURS<49:
        print('creating LP file')
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%Hh%M")
        prob.writeLP(path+'/stanislawLP.lp')
    file = open(path+'/meta_data'+str(start_hour)+'_'+str(end_hour-1)+'.txt', "w")
    file.write('Information concerning this particular LP\n')
    string_to_write = 'hours considered: ['+str(start_hour)+ ','+str(end_hour-1)+ '], including '+str(end_hour-1)+'\n'
    file.write(string_to_write)
    file.write('Constants:\n')
    for key in constants:
        string_to_write = key+' = '+str(constants[key][0])+' '+str(constants[key][1])+'\n'
        file.write(string_to_write)
    string_tmp = 'Optimized rated power of installation = '+ str(PRATEDSOLAR) + ' kWp\n'
    file.write(string_tmp)
    activated = ''
    for var in pv_activation_vars:
        if pv_activation_vars[var].varValue == 1:
            activated += str(var)+','
    activated += '\n'
    file.write('activated binary PV variables: ')
    file.write(activated)
    string_tmp = 'LpStatus: '+LpStatus[prob.status]+'\n'
    file.write(string_tmp)
    string_tmp = 'optimized battery capacity: '+ str(CAPBAT.varValue) + ' kWh \n'
    file.write(string_tmp)
    string_tmp = 'total cost: ' + str(value(prob.objective))[0:12] + ' CHF\n'
    file.write(string_tmp)
    string_tmp = 'total cost if we only bought from the grid: '+str(sum([CENERGYGRID[hour] * PCONS[hour] for hour in hours_considered]))+'\n'
    file.write(string_tmp)
    string_tmp = 'time to solve LP = ' + str(time_to_solve)+' seconds \n'
    file.write(string_tmp)
    string_tmp = '\n cons is sink\n'
    file.write(string_tmp)
    file.close()
