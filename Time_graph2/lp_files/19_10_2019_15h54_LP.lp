\* Energy flow problem *\
Minimize
OBJ: 120 activation_('supersupersourcePV',_'supersourcePV0')
 + 120 activation_('supersupersourcePV',_'supersourcePV1')
 + 120 activation_('supersupersourcePV',_'supersourcePV2')
 + 120 activation_('supersupersourcePV',_'supersourcePV3')
 + 120 activation_('supersupersourcePV',_'supersourcePV4')
 + 120 activation_('supersupersourcePV',_'supersourcePV5')
 + 120 activation_('supersupersourcePV',_'supersourcePV6')
 + 120 activation_('supersupersourcePV',_'supersourcePV7')
 + 120 activation_('supersupersourcePV',_'supersourcePV8')
 + 120 activation_('supersupersourcePV',_'supersourcePV9') + 4 cap_bat
 - 0.05 flow_('PV0',_'supersink') - 0.05 flow_('PV1',_'supersink')
 - 0.05 flow_('PV2',_'supersink') - 0.05 flow_('PV3',_'supersink')
 + 0.2 flow_('supersource',_'Consumption0')
 + 0.2 flow_('supersource',_'Consumption1')
 + 0.2 flow_('supersource',_'Consumption2')
 + 0.2 flow_('supersource',_'Consumption3') + 100 non_zero_pv
Subject To
_C1: flow_('PV0',_'supersink') <= 10
_C10: flow_('supersource',_'Consumption0') >= 0
_C100: flow_('Consumption1',_'supersink') >= 0.408
_C101: flow_('Consumption2',_'supersink') <= 1.472
_C102: flow_('Consumption2',_'supersink') >= 0.416
_C103: flow_('Consumption3',_'supersink') <= 1.472
_C104: flow_('Consumption3',_'supersink') >= 0.472
_C105: - flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C106: flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C107: - activation_('supersupersourcePV',_'supersourcePV0') + non_zero_pv
 >= 0
_C108: flow_('supersupersourcePV',_'supersourcePV1') <= 0
_C109: - activation_('supersupersourcePV',_'supersourcePV1') + non_zero_pv
 >= 0
_C11: flow_('supersource',_'Consumption1') <= 10
_C110: flow_('supersupersourcePV',_'supersourcePV2') <= 0
_C111: - activation_('supersupersourcePV',_'supersourcePV2') + non_zero_pv
 >= 0
_C112: flow_('supersupersourcePV',_'supersourcePV3') <= 0
_C113: - activation_('supersupersourcePV',_'supersourcePV3') + non_zero_pv
 >= 0
_C114: flow_('supersupersourcePV',_'supersourcePV4') <= 0
_C115: - activation_('supersupersourcePV',_'supersourcePV4') + non_zero_pv
 >= 0
_C116: flow_('supersupersourcePV',_'supersourcePV5') <= 0
_C117: - activation_('supersupersourcePV',_'supersourcePV5') + non_zero_pv
 >= 0
_C118: flow_('supersupersourcePV',_'supersourcePV6') <= 0
_C119: - activation_('supersupersourcePV',_'supersourcePV6') + non_zero_pv
 >= 0
_C12: flow_('supersource',_'Consumption1') >= 0
_C120: flow_('supersupersourcePV',_'supersourcePV7') <= 0
_C121: - activation_('supersupersourcePV',_'supersourcePV7') + non_zero_pv
 >= 0
_C122: flow_('supersupersourcePV',_'supersourcePV8') <= 0
_C123: - activation_('supersupersourcePV',_'supersourcePV8') + non_zero_pv
 >= 0
_C124: flow_('supersupersourcePV',_'supersourcePV9') <= 0
_C125: - activation_('supersupersourcePV',_'supersourcePV9') + non_zero_pv
 >= 0
_C126: - flow_('supersourcePV0',_'PV0') - flow_('supersourcePV0',_'PV1')
 - flow_('supersourcePV0',_'PV2') - flow_('supersourcePV0',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV0') = 0
_C127: - flow_('supersourcePV1',_'PV0') - flow_('supersourcePV1',_'PV1')
 - flow_('supersourcePV1',_'PV2') - flow_('supersourcePV1',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV1') = 0
_C128: - flow_('supersourcePV2',_'PV0') - flow_('supersourcePV2',_'PV1')
 - flow_('supersourcePV2',_'PV2') - flow_('supersourcePV2',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV2') = 0
_C129: - flow_('supersourcePV3',_'PV0') - flow_('supersourcePV3',_'PV1')
 - flow_('supersourcePV3',_'PV2') - flow_('supersourcePV3',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV3') = 0
_C13: flow_('supersource',_'Consumption2') <= 10
_C130: - flow_('supersourcePV4',_'PV0') - flow_('supersourcePV4',_'PV1')
 - flow_('supersourcePV4',_'PV2') - flow_('supersourcePV4',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV4') = 0
_C131: - flow_('supersourcePV5',_'PV0') - flow_('supersourcePV5',_'PV1')
 - flow_('supersourcePV5',_'PV2') - flow_('supersourcePV5',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV5') = 0
_C132: - flow_('supersourcePV6',_'PV0') - flow_('supersourcePV6',_'PV1')
 - flow_('supersourcePV6',_'PV2') - flow_('supersourcePV6',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV6') = 0
_C133: - flow_('supersourcePV7',_'PV0') - flow_('supersourcePV7',_'PV1')
 - flow_('supersourcePV7',_'PV2') - flow_('supersourcePV7',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV7') = 0
_C134: - flow_('supersourcePV8',_'PV0') - flow_('supersourcePV8',_'PV1')
 - flow_('supersourcePV8',_'PV2') - flow_('supersourcePV8',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV8') = 0
_C135: - flow_('supersourcePV9',_'PV0') - flow_('supersourcePV9',_'PV1')
 - flow_('supersourcePV9',_'PV2') - flow_('supersourcePV9',_'PV3')
 + flow_('supersupersourcePV',_'supersourcePV9') = 0
_C136: - flow_('PV0',_'Battery0') - flow_('PV0',_'Consumption0')
 - flow_('PV0',_'supersink') + flow_('supersourcePV0',_'PV0')
 + flow_('supersourcePV1',_'PV0') + flow_('supersourcePV2',_'PV0')
 + flow_('supersourcePV3',_'PV0') + flow_('supersourcePV4',_'PV0')
 + flow_('supersourcePV5',_'PV0') + flow_('supersourcePV6',_'PV0')
 + flow_('supersourcePV7',_'PV0') + flow_('supersourcePV8',_'PV0')
 + flow_('supersourcePV9',_'PV0') = 0
_C137: - flow_('PV1',_'Battery1') - flow_('PV1',_'Consumption1')
 - flow_('PV1',_'supersink') + flow_('supersourcePV0',_'PV1')
 + flow_('supersourcePV1',_'PV1') + flow_('supersourcePV2',_'PV1')
 + flow_('supersourcePV3',_'PV1') + flow_('supersourcePV4',_'PV1')
 + flow_('supersourcePV5',_'PV1') + flow_('supersourcePV6',_'PV1')
 + flow_('supersourcePV7',_'PV1') + flow_('supersourcePV8',_'PV1')
 + flow_('supersourcePV9',_'PV1') = 0
_C138: - flow_('PV2',_'Battery2') - flow_('PV2',_'Consumption2')
 - flow_('PV2',_'supersink') + flow_('supersourcePV0',_'PV2')
 + flow_('supersourcePV1',_'PV2') + flow_('supersourcePV2',_'PV2')
 + flow_('supersourcePV3',_'PV2') + flow_('supersourcePV4',_'PV2')
 + flow_('supersourcePV5',_'PV2') + flow_('supersourcePV6',_'PV2')
 + flow_('supersourcePV7',_'PV2') + flow_('supersourcePV8',_'PV2')
 + flow_('supersourcePV9',_'PV2') = 0
_C139: - flow_('PV3',_'Battery3') - flow_('PV3',_'Consumption3')
 - flow_('PV3',_'supersink') + flow_('supersourcePV0',_'PV3')
 + flow_('supersourcePV1',_'PV3') + flow_('supersourcePV2',_'PV3')
 + flow_('supersourcePV3',_'PV3') + flow_('supersourcePV4',_'PV3')
 + flow_('supersourcePV5',_'PV3') + flow_('supersourcePV6',_'PV3')
 + flow_('supersourcePV7',_'PV3') + flow_('supersourcePV8',_'PV3')
 + flow_('supersourcePV9',_'PV3') = 0
_C14: flow_('supersource',_'Consumption2') >= 0
_C140: - flow_('Battery0',_'Battery1') - flow_('Battery0',_'Consumption0')
 + flow_('PV0',_'Battery0') = 0
_C141: flow_('Battery0',_'Battery1') - flow_('Battery1',_'Battery2')
 - flow_('Battery1',_'Consumption1') + flow_('PV1',_'Battery1') = 0
_C142: flow_('Battery1',_'Battery2') - flow_('Battery2',_'Battery3')
 - flow_('Battery2',_'Consumption2') + flow_('PV2',_'Battery2') = 0
_C143: flow_('Battery2',_'Battery3') - flow_('Battery3',_'Consumption3')
 + flow_('PV3',_'Battery3') = 0
_C144: flow_('Battery0',_'Consumption0') - flow_('Consumption0',_'supersink')
 + flow_('PV0',_'Consumption0') + flow_('supersource',_'Consumption0') = 0
_C145: flow_('Battery1',_'Consumption1') - flow_('Consumption1',_'supersink')
 + flow_('PV1',_'Consumption1') + flow_('supersource',_'Consumption1') = 0
_C146: flow_('Battery2',_'Consumption2') - flow_('Consumption2',_'supersink')
 + flow_('PV2',_'Consumption2') + flow_('supersource',_'Consumption2') = 0
_C147: flow_('Battery3',_'Consumption3') - flow_('Consumption3',_'supersink')
 + flow_('PV3',_'Consumption3') + flow_('supersource',_'Consumption3') = 0
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
_C41: flow_('supersourcePV3',_'PV0') <= 0
_C42: flow_('supersourcePV3',_'PV0') >= 0
_C43: flow_('supersourcePV3',_'PV1') <= 0
_C44: flow_('supersourcePV3',_'PV1') >= 0
_C45: flow_('supersourcePV3',_'PV2') <= 0
_C46: flow_('supersourcePV3',_'PV2') >= 0
_C47: flow_('supersourcePV3',_'PV3') <= 0
_C48: flow_('supersourcePV3',_'PV3') >= 0
_C49: flow_('supersourcePV4',_'PV0') <= 0
_C5: flow_('PV2',_'supersink') <= 10
_C50: flow_('supersourcePV4',_'PV0') >= 0
_C51: flow_('supersourcePV4',_'PV1') <= 0
_C52: flow_('supersourcePV4',_'PV1') >= 0
_C53: flow_('supersourcePV4',_'PV2') <= 0
_C54: flow_('supersourcePV4',_'PV2') >= 0
_C55: flow_('supersourcePV4',_'PV3') <= 0
_C56: flow_('supersourcePV4',_'PV3') >= 0
_C57: flow_('supersourcePV5',_'PV0') <= 0
_C58: flow_('supersourcePV5',_'PV0') >= 0
_C59: flow_('supersourcePV5',_'PV1') <= 0
_C6: flow_('PV2',_'supersink') >= 0
_C60: flow_('supersourcePV5',_'PV1') >= 0
_C61: flow_('supersourcePV5',_'PV2') <= 0
_C62: flow_('supersourcePV5',_'PV2') >= 0
_C63: flow_('supersourcePV5',_'PV3') <= 0
_C64: flow_('supersourcePV5',_'PV3') >= 0
_C65: flow_('supersourcePV6',_'PV0') <= 0
_C66: flow_('supersourcePV6',_'PV0') >= 0
_C67: flow_('supersourcePV6',_'PV1') <= 0
_C68: flow_('supersourcePV6',_'PV1') >= 0
_C69: flow_('supersourcePV6',_'PV2') <= 0
_C7: flow_('PV3',_'supersink') <= 10
_C70: flow_('supersourcePV6',_'PV2') >= 0
_C71: flow_('supersourcePV6',_'PV3') <= 0
_C72: flow_('supersourcePV6',_'PV3') >= 0
_C73: flow_('supersourcePV7',_'PV0') <= 0
_C74: flow_('supersourcePV7',_'PV0') >= 0
_C75: flow_('supersourcePV7',_'PV1') <= 0
_C76: flow_('supersourcePV7',_'PV1') >= 0
_C77: flow_('supersourcePV7',_'PV2') <= 0
_C78: flow_('supersourcePV7',_'PV2') >= 0
_C79: flow_('supersourcePV7',_'PV3') <= 0
_C8: flow_('PV3',_'supersink') >= 0
_C80: flow_('supersourcePV7',_'PV3') >= 0
_C81: flow_('supersourcePV8',_'PV0') <= 0
_C82: flow_('supersourcePV8',_'PV0') >= 0
_C83: flow_('supersourcePV8',_'PV1') <= 0
_C84: flow_('supersourcePV8',_'PV1') >= 0
_C85: flow_('supersourcePV8',_'PV2') <= 0
_C86: flow_('supersourcePV8',_'PV2') >= 0
_C87: flow_('supersourcePV8',_'PV3') <= 0
_C88: flow_('supersourcePV8',_'PV3') >= 0
_C89: flow_('supersourcePV9',_'PV0') <= 0
_C9: flow_('supersource',_'Consumption0') <= 10
_C90: flow_('supersourcePV9',_'PV0') >= 0
_C91: flow_('supersourcePV9',_'PV1') <= 0
_C92: flow_('supersourcePV9',_'PV1') >= 0
_C93: flow_('supersourcePV9',_'PV2') <= 0
_C94: flow_('supersourcePV9',_'PV2') >= 0
_C95: flow_('supersourcePV9',_'PV3') <= 0
_C96: flow_('supersourcePV9',_'PV3') >= 0
_C97: flow_('Consumption0',_'supersink') <= 1.472
_C98: flow_('Consumption0',_'supersink') >= 0
_C99: flow_('Consumption1',_'supersink') <= 1.472
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
activation_('supersupersourcePV',_'supersourcePV3')
activation_('supersupersourcePV',_'supersourcePV4')
activation_('supersupersourcePV',_'supersourcePV5')
activation_('supersupersourcePV',_'supersourcePV6')
activation_('supersupersourcePV',_'supersourcePV7')
activation_('supersupersourcePV',_'supersourcePV8')
activation_('supersupersourcePV',_'supersourcePV9')
non_zero_pv
End
