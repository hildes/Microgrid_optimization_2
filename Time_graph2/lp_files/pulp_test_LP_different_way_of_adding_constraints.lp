\* Min cost flow problem *\
Minimize
Total_Cost_of_flow: - flow_('B',_'supersink') - 2 flow_('C',_'supersink')
Subject To
_C1: flow_('supersource',_'A') <= 5
_C10: flow_('B',_'C') >= 0
_C11: flow_('A',_'supersink') <= 9
_C12: flow_('A',_'supersink') >= 0
_C13: flow_('B',_'supersink') <= 5
_C14: flow_('B',_'supersink') >= 0
_C15: flow_('C',_'supersink') <= 6
_C16: flow_('C',_'supersink') >= 0
_C17: - flow_('A',_'C') - flow_('A',_'supersink') + flow_('supersource',_'A')
 = 0
_C18: - flow_('B',_'C') - flow_('B',_'supersink') + flow_('supersource',_'B')
 = 0
_C19: flow_('A',_'C') + flow_('B',_'C') - flow_('C',_'supersink')
 + flow_('supersource',_'C') = 0
_C2: flow_('supersource',_'A') >= 0
_C3: flow_('supersource',_'B') <= 5
_C4: flow_('supersource',_'B') >= 0
_C5: flow_('supersource',_'C') <= 5
_C6: flow_('supersource',_'C') >= 0
_C7: flow_('A',_'C') <= 3.14
_C8: flow_('A',_'C') >= 0
_C9: flow_('B',_'C') <= 2.1
Bounds
flow_('A',_'C') free
flow_('A',_'supersink') free
flow_('B',_'C') free
flow_('B',_'supersink') free
flow_('C',_'supersink') free
flow_('supersource',_'A') free
flow_('supersource',_'B') free
flow_('supersource',_'C') free
End
