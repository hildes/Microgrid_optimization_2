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
 - 0.05 flow_('PV10',_'supersink') - 0.05 flow_('PV11',_'supersink')
 - 0.05 flow_('PV9',_'supersink') + 0.2 flow_('supersource',_'Consumption10')
 + 0.2 flow_('supersource',_'Consumption11')
 + 0.2 flow_('supersource',_'Consumption9') + 10000 non_zero_pv
Subject To
_C1: flow_('PV9',_'supersink') <= 10
_C10: flow_('supersource',_'Consumption10') >= 0
_C100: - flow_('supersourcePV0',_'PV10') - flow_('supersourcePV0',_'PV11')
 - flow_('supersourcePV0',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV0') = 0
_C101: - flow_('supersourcePV1',_'PV10') - flow_('supersourcePV1',_'PV11')
 - flow_('supersourcePV1',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV1') = 0
_C102: - flow_('supersourcePV2',_'PV10') - flow_('supersourcePV2',_'PV11')
 - flow_('supersourcePV2',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV2') = 0
_C103: - flow_('supersourcePV3',_'PV10') - flow_('supersourcePV3',_'PV11')
 - flow_('supersourcePV3',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV3') = 0
_C104: - flow_('supersourcePV4',_'PV10') - flow_('supersourcePV4',_'PV11')
 - flow_('supersourcePV4',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV4') = 0
_C105: - flow_('supersourcePV5',_'PV10') - flow_('supersourcePV5',_'PV11')
 - flow_('supersourcePV5',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV5') = 0
_C106: - flow_('supersourcePV6',_'PV10') - flow_('supersourcePV6',_'PV11')
 - flow_('supersourcePV6',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV6') = 0
_C107: - flow_('supersourcePV7',_'PV10') - flow_('supersourcePV7',_'PV11')
 - flow_('supersourcePV7',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV7') = 0
_C108: - flow_('supersourcePV8',_'PV10') - flow_('supersourcePV8',_'PV11')
 - flow_('supersourcePV8',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV8') = 0
_C109: - flow_('supersourcePV9',_'PV10') - flow_('supersourcePV9',_'PV11')
 - flow_('supersourcePV9',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV9') = 0
_C11: flow_('supersource',_'Consumption11') <= 100
_C110: - flow_('PV9',_'Battery9') - flow_('PV9',_'Consumption9')
 - flow_('PV9',_'supersink') + flow_('supersourcePV0',_'PV9')
 + flow_('supersourcePV1',_'PV9') + flow_('supersourcePV2',_'PV9')
 + flow_('supersourcePV3',_'PV9') + flow_('supersourcePV4',_'PV9')
 + flow_('supersourcePV5',_'PV9') + flow_('supersourcePV6',_'PV9')
 + flow_('supersourcePV7',_'PV9') + flow_('supersourcePV8',_'PV9')
 + flow_('supersourcePV9',_'PV9') = 0
_C111: - flow_('PV10',_'Battery10') - flow_('PV10',_'Consumption10')
 - flow_('PV10',_'supersink') + flow_('supersourcePV0',_'PV10')
 + flow_('supersourcePV1',_'PV10') + flow_('supersourcePV2',_'PV10')
 + flow_('supersourcePV3',_'PV10') + flow_('supersourcePV4',_'PV10')
 + flow_('supersourcePV5',_'PV10') + flow_('supersourcePV6',_'PV10')
 + flow_('supersourcePV7',_'PV10') + flow_('supersourcePV8',_'PV10')
 + flow_('supersourcePV9',_'PV10') = 0
_C112: - flow_('PV11',_'Battery11') - flow_('PV11',_'Consumption11')
 - flow_('PV11',_'supersink') + flow_('supersourcePV0',_'PV11')
 + flow_('supersourcePV1',_'PV11') + flow_('supersourcePV2',_'PV11')
 + flow_('supersourcePV3',_'PV11') + flow_('supersourcePV4',_'PV11')
 + flow_('supersourcePV5',_'PV11') + flow_('supersourcePV6',_'PV11')
 + flow_('supersourcePV7',_'PV11') + flow_('supersourcePV8',_'PV11')
 + flow_('supersourcePV9',_'PV11') = 0
_C113: - flow_('Battery9',_'Battery10') - flow_('Battery9',_'Consumption9')
 + flow_('PV9',_'Battery9') = 0
_C114: - flow_('Battery10',_'Battery11') - flow_('Battery10',_'Consumption10')
 + flow_('Battery9',_'Battery10') + flow_('PV10',_'Battery10') = 0
_C115: flow_('Battery10',_'Battery11') - flow_('Battery11',_'Consumption11')
 + flow_('PV11',_'Battery11') = 0
_C116: flow_('Battery9',_'Consumption9') - flow_('Consumption9',_'supersink')
 + flow_('PV9',_'Consumption9') + flow_('supersource',_'Consumption9') = 0
_C117: flow_('Battery10',_'Consumption10')
 - flow_('Consumption10',_'supersink') + flow_('PV10',_'Consumption10')
 + flow_('supersource',_'Consumption10') = 0
_C118: flow_('Battery11',_'Consumption11')
 - flow_('Consumption11',_'supersink') + flow_('PV11',_'Consumption11')
 + flow_('supersource',_'Consumption11') = 0
_C12: flow_('supersource',_'Consumption11') >= 0
_C13: flow_('supersourcePV0',_'PV9') <= 0.0766902360387
_C14: flow_('supersourcePV0',_'PV9') >= 0
_C15: flow_('supersourcePV0',_'PV10') <= 0.129477357356
_C16: flow_('supersourcePV0',_'PV10') >= 0
_C17: flow_('supersourcePV0',_'PV11') <= 0.219984721858
_C18: flow_('supersourcePV0',_'PV11') >= 0
_C19: flow_('supersourcePV1',_'PV9') <= 0.0766902360387
_C2: flow_('PV9',_'supersink') >= 0
_C20: flow_('supersourcePV1',_'PV9') >= 0
_C21: flow_('supersourcePV1',_'PV10') <= 0.129477357356
_C22: flow_('supersourcePV1',_'PV10') >= 0
_C23: flow_('supersourcePV1',_'PV11') <= 0.219984721858
_C24: flow_('supersourcePV1',_'PV11') >= 0
_C25: flow_('supersourcePV2',_'PV9') <= 0.0766902360387
_C26: flow_('supersourcePV2',_'PV9') >= 0
_C27: flow_('supersourcePV2',_'PV10') <= 0.129477357356
_C28: flow_('supersourcePV2',_'PV10') >= 0
_C29: flow_('supersourcePV2',_'PV11') <= 0.219984721858
_C3: flow_('PV10',_'supersink') <= 10
_C30: flow_('supersourcePV2',_'PV11') >= 0
_C31: flow_('supersourcePV3',_'PV9') <= 0.0766902360387
_C32: flow_('supersourcePV3',_'PV9') >= 0
_C33: flow_('supersourcePV3',_'PV10') <= 0.129477357356
_C34: flow_('supersourcePV3',_'PV10') >= 0
_C35: flow_('supersourcePV3',_'PV11') <= 0.219984721858
_C36: flow_('supersourcePV3',_'PV11') >= 0
_C37: flow_('supersourcePV4',_'PV9') <= 0.0766902360387
_C38: flow_('supersourcePV4',_'PV9') >= 0
_C39: flow_('supersourcePV4',_'PV10') <= 0.129477357356
_C4: flow_('PV10',_'supersink') >= 0
_C40: flow_('supersourcePV4',_'PV10') >= 0
_C41: flow_('supersourcePV4',_'PV11') <= 0.219984721858
_C42: flow_('supersourcePV4',_'PV11') >= 0
_C43: flow_('supersourcePV5',_'PV9') <= 0.0766902360387
_C44: flow_('supersourcePV5',_'PV9') >= 0
_C45: flow_('supersourcePV5',_'PV10') <= 0.129477357356
_C46: flow_('supersourcePV5',_'PV10') >= 0
_C47: flow_('supersourcePV5',_'PV11') <= 0.219984721858
_C48: flow_('supersourcePV5',_'PV11') >= 0
_C49: flow_('supersourcePV6',_'PV9') <= 0.0766902360387
_C5: flow_('PV11',_'supersink') <= 10
_C50: flow_('supersourcePV6',_'PV9') >= 0
_C51: flow_('supersourcePV6',_'PV10') <= 0.129477357356
_C52: flow_('supersourcePV6',_'PV10') >= 0
_C53: flow_('supersourcePV6',_'PV11') <= 0.219984721858
_C54: flow_('supersourcePV6',_'PV11') >= 0
_C55: flow_('supersourcePV7',_'PV9') <= 0.0766902360387
_C56: flow_('supersourcePV7',_'PV9') >= 0
_C57: flow_('supersourcePV7',_'PV10') <= 0.129477357356
_C58: flow_('supersourcePV7',_'PV10') >= 0
_C59: flow_('supersourcePV7',_'PV11') <= 0.219984721858
_C6: flow_('PV11',_'supersink') >= 0
_C60: flow_('supersourcePV7',_'PV11') >= 0
_C61: flow_('supersourcePV8',_'PV9') <= 0.0766902360387
_C62: flow_('supersourcePV8',_'PV9') >= 0
_C63: flow_('supersourcePV8',_'PV10') <= 0.129477357356
_C64: flow_('supersourcePV8',_'PV10') >= 0
_C65: flow_('supersourcePV8',_'PV11') <= 0.219984721858
_C66: flow_('supersourcePV8',_'PV11') >= 0
_C67: flow_('supersourcePV9',_'PV9') <= 0.0766902360387
_C68: flow_('supersourcePV9',_'PV9') >= 0
_C69: flow_('supersourcePV9',_'PV10') <= 0.129477357356
_C7: flow_('supersource',_'Consumption9') <= 100
_C70: flow_('supersourcePV9',_'PV10') >= 0
_C71: flow_('supersourcePV9',_'PV11') <= 0.219984721858
_C72: flow_('supersourcePV9',_'PV11') >= 0
_C73: flow_('Consumption9',_'supersink') <= 1.00184
_C74: flow_('Consumption9',_'supersink') >= 0
_C75: flow_('Consumption10',_'supersink') <= 1.00184
_C76: flow_('Consumption10',_'supersink') >= 0.001832
_C77: flow_('Consumption11',_'supersink') <= 1.00184
_C78: flow_('Consumption11',_'supersink') >= 0.00184
_C79: 0.426152315252 activation_('supersupersourcePV',_'supersourcePV0')
 - flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C8: flow_('supersource',_'Consumption9') >= 0
_C80: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV0')
 + flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C81: - activation_('supersupersourcePV',_'supersourcePV0') + non_zero_pv >= 0
_C82: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV1')
 + flow_('supersupersourcePV',_'supersourcePV1') <= 0
_C83: - activation_('supersupersourcePV',_'supersourcePV1') + non_zero_pv >= 0
_C84: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV2')
 + flow_('supersupersourcePV',_'supersourcePV2') <= 0
_C85: - activation_('supersupersourcePV',_'supersourcePV2') + non_zero_pv >= 0
_C86: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV3')
 + flow_('supersupersourcePV',_'supersourcePV3') <= 0
_C87: - activation_('supersupersourcePV',_'supersourcePV3') + non_zero_pv >= 0
_C88: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV4')
 + flow_('supersupersourcePV',_'supersourcePV4') <= 0
_C89: - activation_('supersupersourcePV',_'supersourcePV4') + non_zero_pv >= 0
_C9: flow_('supersource',_'Consumption10') <= 100
_C90: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV5')
 + flow_('supersupersourcePV',_'supersourcePV5') <= 0
_C91: - activation_('supersupersourcePV',_'supersourcePV5') + non_zero_pv >= 0
_C92: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV6')
 + flow_('supersupersourcePV',_'supersourcePV6') <= 0
_C93: - activation_('supersupersourcePV',_'supersourcePV6') + non_zero_pv >= 0
_C94: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV7')
 + flow_('supersupersourcePV',_'supersourcePV7') <= 0
_C95: - activation_('supersupersourcePV',_'supersourcePV7') + non_zero_pv >= 0
_C96: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV8')
 + flow_('supersupersourcePV',_'supersourcePV8') <= 0
_C97: - activation_('supersupersourcePV',_'supersourcePV8') + non_zero_pv >= 0
_C98: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV9')
 + flow_('supersupersourcePV',_'supersourcePV9') <= 0
_C99: - activation_('supersupersourcePV',_'supersourcePV9') + non_zero_pv >= 0
battery_capacityBattery10: - cap_bat + flow_('Battery10',_'Battery11') <= 0
battery_capacityBattery9: - cap_bat + flow_('Battery9',_'Battery10') <= 0
charging_ratePV10: - cap_bat + flow_('PV10',_'Battery10') <= 0
charging_ratePV11: - cap_bat + flow_('PV11',_'Battery11') <= 0
charging_ratePV9: - cap_bat + flow_('PV9',_'Battery9') <= 0
discharging_rateBattery10: - cap_bat + flow_('Battery10',_'Consumption10')
 <= 0
discharging_rateBattery11: - cap_bat + flow_('Battery11',_'Consumption11')
 <= 0
discharging_rateBattery9: - cap_bat + flow_('Battery9',_'Consumption9') <= 0
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
