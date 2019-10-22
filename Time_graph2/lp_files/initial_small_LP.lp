\* Min cost flow problem *\
Minimize
Total_Cost_of_flow: - flow_('B',_'supersink') - 2 flow_('C',_'supersink')
Subject To
_C1: - flow_('A',_'C') - flow_('A',_'supersink') + flow_('supersource',_'A')
 = 0
_C2: - flow_('B',_'C') - flow_('B',_'supersink') + flow_('supersource',_'B')
 = 0
_C3: flow_('A',_'C') + flow_('B',_'C') - flow_('C',_'supersink')
 + flow_('supersource',_'C') = 0
Bounds
flow_('A',_'C') <= 3.14
flow_('A',_'supersink') <= 9
flow_('B',_'C') <= 2.1
flow_('B',_'supersink') <= 5
flow_('C',_'supersink') <= 6
flow_('supersource',_'A') <= 5
flow_('supersource',_'B') <= 5
flow_('supersource',_'C') <= 5
End
