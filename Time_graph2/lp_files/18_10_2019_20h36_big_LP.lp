\* Energy flow problem *\
Minimize
OBJ: 1200 activation_('supersupersourcePV',_'supersourcePV0')
 + 1200 activation_('supersupersourcePV',_'supersourcePV1')
 + 1200 activation_('supersupersourcePV',_'supersourcePV2')
 + 1200 activation_('supersupersourcePV',_'supersourcePV3')
 + 1200 activation_('supersupersourcePV',_'supersourcePV4')
 + 1200 activation_('supersupersourcePV',_'supersourcePV5')
 + 1200 activation_('supersupersourcePV',_'supersourcePV6')
 + 1200 activation_('supersupersourcePV',_'supersourcePV7')
 + 1200 activation_('supersupersourcePV',_'supersourcePV8')
 + 1200 activation_('supersupersourcePV',_'supersourcePV9') + 470 cap_bat
 - 0.05 flow_('PV0',_'supersink') - 0.05 flow_('PV1',_'supersink')
 - 0.05 flow_('PV2',_'supersink') + 0.2 flow_('supersource',_'Consumption0')
 + 0.2 flow_('supersource',_'Consumption1')
 + 0.2 flow_('supersource',_'Consumption2') + 10000 non_zero_pv
Subject To
_C1: - cap_bat + flow_('Battery0',_'Battery1') <= 0
_C10: flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C11: - activation_('supersupersourcePV',_'supersourcePV0') + non_zero_pv >= 0
_C12: flow_('supersupersourcePV',_'supersourcePV1') <= 0
_C13: - activation_('supersupersourcePV',_'supersourcePV1') + non_zero_pv >= 0
_C14: flow_('supersupersourcePV',_'supersourcePV2') <= 0
_C15: - activation_('supersupersourcePV',_'supersourcePV2') + non_zero_pv >= 0
_C16: flow_('supersupersourcePV',_'supersourcePV3') <= 0
_C17: - activation_('supersupersourcePV',_'supersourcePV3') + non_zero_pv >= 0
_C18: flow_('supersupersourcePV',_'supersourcePV4') <= 0
_C19: - activation_('supersupersourcePV',_'supersourcePV4') + non_zero_pv >= 0
_C2: - cap_bat + flow_('Battery1',_'Battery2') <= 0
_C20: flow_('supersupersourcePV',_'supersourcePV5') <= 0
_C21: - activation_('supersupersourcePV',_'supersourcePV5') + non_zero_pv >= 0
_C22: flow_('supersupersourcePV',_'supersourcePV6') <= 0
_C23: - activation_('supersupersourcePV',_'supersourcePV6') + non_zero_pv >= 0
_C24: flow_('supersupersourcePV',_'supersourcePV7') <= 0
_C25: - activation_('supersupersourcePV',_'supersourcePV7') + non_zero_pv >= 0
_C26: flow_('supersupersourcePV',_'supersourcePV8') <= 0
_C27: - activation_('supersupersourcePV',_'supersourcePV8') + non_zero_pv >= 0
_C28: flow_('supersupersourcePV',_'supersourcePV9') <= 0
_C29: - activation_('supersupersourcePV',_'supersourcePV9') + non_zero_pv >= 0
_C3: - cap_bat + flow_('PV0',_'Battery0') <= 0
_C30: - flow_('supersourcePV0',_'PV0') - flow_('supersourcePV0',_'PV1')
 - flow_('supersourcePV0',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV0') = 0
_C31: - flow_('supersourcePV1',_'PV0') - flow_('supersourcePV1',_'PV1')
 - flow_('supersourcePV1',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV1') = 0
_C32: - flow_('supersourcePV2',_'PV0') - flow_('supersourcePV2',_'PV1')
 - flow_('supersourcePV2',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV2') = 0
_C33: - flow_('supersourcePV3',_'PV0') - flow_('supersourcePV3',_'PV1')
 - flow_('supersourcePV3',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV3') = 0
_C34: - flow_('supersourcePV4',_'PV0') - flow_('supersourcePV4',_'PV1')
 - flow_('supersourcePV4',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV4') = 0
_C35: - flow_('supersourcePV5',_'PV0') - flow_('supersourcePV5',_'PV1')
 - flow_('supersourcePV5',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV5') = 0
_C36: - flow_('supersourcePV6',_'PV0') - flow_('supersourcePV6',_'PV1')
 - flow_('supersourcePV6',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV6') = 0
_C37: - flow_('supersourcePV7',_'PV0') - flow_('supersourcePV7',_'PV1')
 - flow_('supersourcePV7',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV7') = 0
_C38: - flow_('supersourcePV8',_'PV0') - flow_('supersourcePV8',_'PV1')
 - flow_('supersourcePV8',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV8') = 0
_C39: - flow_('supersourcePV9',_'PV0') - flow_('supersourcePV9',_'PV1')
 - flow_('supersourcePV9',_'PV2')
 + flow_('supersupersourcePV',_'supersourcePV9') = 0
_C4: - cap_bat + flow_('PV1',_'Battery1') <= 0
_C40: - flow_('Battery0',_'Battery1') - flow_('Battery0',_'Consumption0')
 + flow_('PV0',_'Battery0') = 0
_C41: flow_('Battery0',_'Battery1') - flow_('Battery1',_'Battery2')
 - flow_('Battery1',_'Consumption1') + flow_('PV1',_'Battery1') = 0
_C42: flow_('Battery1',_'Battery2') - flow_('Battery2',_'Consumption2')
 + flow_('PV2',_'Battery2') = 0
_C5: - cap_bat + flow_('PV2',_'Battery2') <= 0
_C6: - cap_bat + flow_('Battery0',_'Consumption0') <= 0
_C7: - cap_bat + flow_('Battery1',_'Consumption1') <= 0
_C8: - cap_bat + flow_('Battery2',_'Consumption2') <= 0
_C9: - flow_('supersupersourcePV',_'supersourcePV0') <= 0
Bounds
cap_bat <= 10
flow_('PV0',_'supersink') = 10
flow_('PV1',_'supersink') = 10
flow_('PV2',_'supersink') = 10
flow_('supersource',_'Consumption0') = 1000
flow_('supersource',_'Consumption1') = 1000
flow_('supersource',_'Consumption2') = 1000
flow_('supersourcePV0',_'PV0') = 0
flow_('supersourcePV0',_'PV1') = 0
flow_('supersourcePV0',_'PV2') = 0
flow_('supersourcePV1',_'PV0') = 0
flow_('supersourcePV1',_'PV1') = 0
flow_('supersourcePV1',_'PV2') = 0
flow_('supersourcePV2',_'PV0') = 0
flow_('supersourcePV2',_'PV1') = 0
flow_('supersourcePV2',_'PV2') = 0
flow_('supersourcePV3',_'PV0') = 0
flow_('supersourcePV3',_'PV1') = 0
flow_('supersourcePV3',_'PV2') = 0
flow_('supersourcePV4',_'PV0') = 0
flow_('supersourcePV4',_'PV1') = 0
flow_('supersourcePV4',_'PV2') = 0
flow_('supersourcePV5',_'PV0') = 0
flow_('supersourcePV5',_'PV1') = 0
flow_('supersourcePV5',_'PV2') = 0
flow_('supersourcePV6',_'PV0') = 0
flow_('supersourcePV6',_'PV1') = 0
flow_('supersourcePV6',_'PV2') = 0
flow_('supersourcePV7',_'PV0') = 0
flow_('supersourcePV7',_'PV1') = 0
flow_('supersourcePV7',_'PV2') = 0
flow_('supersourcePV8',_'PV0') = 0
flow_('supersourcePV8',_'PV1') = 0
flow_('supersourcePV8',_'PV2') = 0
flow_('supersourcePV9',_'PV0') = 0
flow_('supersourcePV9',_'PV1') = 0
flow_('supersourcePV9',_'PV2') = 0
Binaries
activation_('supersupersourcePV',_'supersourcePV0')
activation_('supersupersourcePV',_'supersourcePV1')
activation_('supersupersourcePV',_'supersourcePV2')
activation_('supersupersourcePV',_'supersourcePV3')
activation_('supersupersourcePV',_'supersourcePV4')
activation_('supersupersourcePV',_'supersourcePV5')
activation_('supersupersourcePV',_'supersourcePV6')
activation_('supersupersourcePV',_'supersourcePV7')
activation_('supersupersourcePV',_'supersourcePV8')
activation_('supersupersourcePV',_'supersourcePV9')
non_zero_pv
End
