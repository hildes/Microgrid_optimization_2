\* Energy flow problem *\
Minimize
OBJ: 12 activation_('supersupersourcePV',_'supersourcePV0')
 + 12 activation_('supersupersourcePV',_'supersourcePV1')
 + 12 activation_('supersupersourcePV',_'supersourcePV2') + 4 cap_bat
 - 0.05 flow_('PV0',_'supersink') - 0.05 flow_('PV1',_'supersink')
 - 0.05 flow_('PV2',_'supersink') - 0.05 flow_('PV3',_'supersink')
 + 0.2 flow_('supersource',_'Consumption0')
 + 0.2 flow_('supersource',_'Consumption1')
 + 0.2 flow_('supersource',_'Consumption2')
 + 0.2 flow_('supersource',_'Consumption3') + 10 non_zero_pv
Subject To
_C1: flow_('PV0',_'supersink') <= 10
_C10: flow_('supersource',_'Consumption0') >= 0
_C11: flow_('supersource',_'Consumption1') <= 10
_C12: flow_('supersource',_'Consumption1') >= 0
_C13: flow_('supersource',_'Consumption2') <= 10
_C14: flow_('supersource',_'Consumption2') >= 0
_C15: flow_('supersource',_'Consumption3') <= 10
_C16: flow_('supersource',_'Consumption3') >= 0
_C17: flow_('supersourcePV0',_'PV0') <= 0
_C18: flow_('supersourcePV0',_'PV0') >= 0
_C19: flow_('supersourcePV0',_'PV1') <= 0
_C2: flow_('PV0',_'supersink') >= 0
_C20: flow_('supersourcePV0',_'PV1') >= 0
_C21: flow_('supersourcePV0',_'PV2') <= 0
_C22: flow_('supersourcePV0',_'PV2') >= 0
_C23: flow_('supersourcePV0',_'PV3') <= 0
_C24: flow_('supersourcePV0',_'PV3') >= 0
_C25: flow_('supersourcePV1',_'PV0') <= 0
_C26: flow_('supersourcePV1',_'PV0') >= 0
_C27: flow_('supersourcePV1',_'PV1') <= 0
_C28: flow_('supersourcePV1',_'PV1') >= 0
_C29: flow_('supersourcePV1',_'PV2') <= 0
_C3: flow_('PV1',_'supersink') <= 10
_C30: flow_('supersourcePV1',_'PV2') >= 0
_C31: flow_('supersourcePV1',_'PV3') <= 0
_C32: flow_('supersourcePV1',_'PV3') >= 0
_C33: flow_('supersourcePV2',_'PV0') <= 0
_C34: flow_('supersourcePV2',_'PV0') >= 0
_C35: flow_('supersourcePV2',_'PV1') <= 0
_C36: flow_('supersourcePV2',_'PV1') >= 0
_C37: flow_('supersourcePV2',_'PV2') <= 0
_C38: flow_('supersourcePV2',_'PV2') >= 0
_C39: flow_('supersourcePV2',_'PV3') <= 0
_C4: flow_('PV1',_'supersink') >= 0
_C40: flow_('supersourcePV2',_'PV3') >= 0
_C41: flow_('Consumption0',_'supersink') <= 1.472
_C42: flow_('Consumption0',_'supersink') >= 0
_C43: flow_('Consumption1',_'supersink') <= 1.472
_C44: flow_('Consumption1',_'supersink') >= 0.408
_C45: flow_('Consumption2',_'supersink') <= 1.472
_C46: flow_('Consumption2',_'supersink') >= 0.416
_C47: flow_('Consumption3',_'supersink') <= 1.472
_C48: flow_('Consumption3',_'supersink') >= 0.472
_C49: - flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C5: flow_('PV2',_'supersink') <= 10
_C50: flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C51: - activation_('supersupersourcePV',_'supersourcePV0') + non_zero_pv >= 0
_C52: flow_('supersupersourcePV',_'supersourcePV1') <= 0
_C53: - activation_('supersupersourcePV',_'supersourcePV1') + non_zero_pv >= 0
_C54: flow_('supersupersourcePV',_'supersourcePV2') <= 0
_C55: - activation_('supersupersourcePV',_'supersourcePV2') + non_zero_pv >= 0
_C56: - flow_('supersourcePV0',_'PV0') - flow_('supersourcePV0',_'PV1')
 - flow_('supersourcePV0',_'PV2') - flow_('supersourcePV0',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV0') = 0
_C57: - flow_('supersourcePV1',_'PV0') - flow_('supersourcePV1',_'PV1')
 - flow_('supersourcePV1',_'PV2') - flow_('supersourcePV1',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV1') = 0
_C58: - flow_('supersourcePV2',_'PV0') - flow_('supersourcePV2',_'PV1')
 - flow_('supersourcePV2',_'PV2') - flow_('supersourcePV2',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV2') = 0
_C59: - flow_('PV0',_'Battery0') - flow_('PV0',_'Consumption0')
 - flow_('PV0',_'supersink') + flow_('supersourcePV0',_'PV0')
 + flow_('supersourcePV1',_'PV0') + flow_('supersourcePV2',_'PV0') = 0
_C6: flow_('PV2',_'supersink') >= 0
_C60: - flow_('PV1',_'Battery1') - flow_('PV1',_'Consumption1')
 - flow_('PV1',_'supersink') + flow_('supersourcePV0',_'PV1')
 + flow_('supersourcePV1',_'PV1') + flow_('supersourcePV2',_'PV1') = 0
_C61: - flow_('PV2',_'Battery2') - flow_('PV2',_'Consumption2')
 - flow_('PV2',_'supersink') + flow_('supersourcePV0',_'PV2')
 + flow_('supersourcePV1',_'PV2') + flow_('supersourcePV2',_'PV2') = 0
_C62: - flow_('PV3',_'Battery3') - flow_('PV3',_'Consumption3')
 - flow_('PV3',_'supersink') + flow_('supersourcePV0',_'PV3')
 + flow_('supersourcePV1',_'PV3') + flow_('supersourcePV2',_'PV3') = 0
_C63: - flow_('Battery0',_'Battery1') - flow_('Battery0',_'Consumption0')
 + flow_('PV0',_'Battery0') = 0
_C64: flow_('Battery0',_'Battery1') - flow_('Battery1',_'Battery2')
 - flow_('Battery1',_'Consumption1') + flow_('PV1',_'Battery1') = 0
_C65: flow_('Battery1',_'Battery2') - flow_('Battery2',_'Battery3')
 - flow_('Battery2',_'Consumption2') + flow_('PV2',_'Battery2') = 0
_C66: flow_('Battery2',_'Battery3') - flow_('Battery3',_'Consumption3')
 + flow_('PV3',_'Battery3') = 0
_C67: flow_('Battery0',_'Consumption0') - flow_('Consumption0',_'supersink')
 + flow_('PV0',_'Consumption0') + flow_('supersource',_'Consumption0') = 0
_C68: flow_('Battery1',_'Consumption1') - flow_('Consumption1',_'supersink')
 + flow_('PV1',_'Consumption1') + flow_('supersource',_'Consumption1') = 0
_C69: flow_('Battery2',_'Consumption2') - flow_('Consumption2',_'supersink')
 + flow_('PV2',_'Consumption2') + flow_('supersource',_'Consumption2') = 0
_C7: flow_('PV3',_'supersink') <= 10
_C70: flow_('Battery3',_'Consumption3') - flow_('Consumption3',_'supersink')
 + flow_('PV3',_'Consumption3') + flow_('supersource',_'Consumption3') = 0
_C8: flow_('PV3',_'supersink') >= 0
_C9: flow_('supersource',_'Consumption0') <= 10
battery_capacityBattery0: - cap_bat + flow_('Battery0',_'Battery1') <= 0
battery_capacityBattery1: - cap_bat + flow_('Battery1',_'Battery2') <= 0
battery_capacityBattery2: - cap_bat + flow_('Battery2',_'Battery3') <= 0
charging_ratePV0: - cap_bat + flow_('PV0',_'Battery0') <= 0
charging_ratePV1: - cap_bat + flow_('PV1',_'Battery1') <= 0
charging_ratePV2: - cap_bat + flow_('PV2',_'Battery2') <= 0
charging_ratePV3: - cap_bat + flow_('PV3',_'Battery3') <= 0
discharging_rateBattery0: - cap_bat + flow_('Battery0',_'Consumption0') <= 0
discharging_rateBattery1: - cap_bat + flow_('Battery1',_'Consumption1') <= 0
discharging_rateBattery2: - cap_bat + flow_('Battery2',_'Consumption2') <= 0
discharging_rateBattery3: - cap_bat + flow_('Battery3',_'Consumption3') <= 0
Bounds
cap_bat <= 10
Binaries
activation_('supersupersourcePV',_'supersourcePV0')
activation_('supersupersourcePV',_'supersourcePV1')
activation_('supersupersourcePV',_'supersourcePV2')
non_zero_pv
End
