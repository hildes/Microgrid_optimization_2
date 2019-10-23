\* Energy flow problem *\
Minimize
OBJ: 12 activation_('supersupersourcePV',_'supersourcePV0')
 + 12 activation_('supersupersourcePV',_'supersourcePV1')
 + 12 activation_('supersupersourcePV',_'supersourcePV10')
 + 12 activation_('supersupersourcePV',_'supersourcePV11')
 + 12 activation_('supersupersourcePV',_'supersourcePV12')
 + 12 activation_('supersupersourcePV',_'supersourcePV13')
 + 12 activation_('supersupersourcePV',_'supersourcePV14')
 + 12 activation_('supersupersourcePV',_'supersourcePV15')
 + 12 activation_('supersupersourcePV',_'supersourcePV16')
 + 12 activation_('supersupersourcePV',_'supersourcePV17')
 + 12 activation_('supersupersourcePV',_'supersourcePV18')
 + 12 activation_('supersupersourcePV',_'supersourcePV19')
 + 12 activation_('supersupersourcePV',_'supersourcePV2')
 + 12 activation_('supersupersourcePV',_'supersourcePV20')
 + 12 activation_('supersupersourcePV',_'supersourcePV21')
 + 12 activation_('supersupersourcePV',_'supersourcePV22')
 + 12 activation_('supersupersourcePV',_'supersourcePV23')
 + 12 activation_('supersupersourcePV',_'supersourcePV24')
 + 12 activation_('supersupersourcePV',_'supersourcePV3')
 + 12 activation_('supersupersourcePV',_'supersourcePV4')
 + 12 activation_('supersupersourcePV',_'supersourcePV5')
 + 12 activation_('supersupersourcePV',_'supersourcePV6')
 + 12 activation_('supersupersourcePV',_'supersourcePV7')
 + 12 activation_('supersupersourcePV',_'supersourcePV8')
 + 12 activation_('supersupersourcePV',_'supersourcePV9') + 9 cap_bat
 - 0.05 flow_('PV0',_'supersink') - 0.05 flow_('PV1',_'supersink')
 + 0.2 flow_('supersource',_'Consumption0')
 + 0.2 flow_('supersource',_'Consumption1') + 13 non_zero_pv
Subject To
_C1: - flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C10: flow_('supersupersourcePV',_'supersourcePV4') <= 0
_C11: - activation_('supersupersourcePV',_'supersourcePV4') + non_zero_pv >= 0
_C12: flow_('supersupersourcePV',_'supersourcePV5') <= 0
_C13: - activation_('supersupersourcePV',_'supersourcePV5') + non_zero_pv >= 0
_C14: flow_('supersupersourcePV',_'supersourcePV6') <= 0
_C15: - activation_('supersupersourcePV',_'supersourcePV6') + non_zero_pv >= 0
_C16: flow_('supersupersourcePV',_'supersourcePV7') <= 0
_C17: - activation_('supersupersourcePV',_'supersourcePV7') + non_zero_pv >= 0
_C18: flow_('supersupersourcePV',_'supersourcePV8') <= 0
_C19: - activation_('supersupersourcePV',_'supersourcePV8') + non_zero_pv >= 0
_C2: flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C20: flow_('supersupersourcePV',_'supersourcePV9') <= 0
_C21: - activation_('supersupersourcePV',_'supersourcePV9') + non_zero_pv >= 0
_C22: flow_('supersupersourcePV',_'supersourcePV10') <= 0
_C23: - activation_('supersupersourcePV',_'supersourcePV10') + non_zero_pv
 >= 0
_C24: flow_('supersupersourcePV',_'supersourcePV11') <= 0
_C25: - activation_('supersupersourcePV',_'supersourcePV11') + non_zero_pv
 >= 0
_C26: flow_('supersupersourcePV',_'supersourcePV12') <= 0
_C27: - activation_('supersupersourcePV',_'supersourcePV12') + non_zero_pv
 >= 0
_C28: flow_('supersupersourcePV',_'supersourcePV13') <= 0
_C29: - activation_('supersupersourcePV',_'supersourcePV13') + non_zero_pv
 >= 0
_C3: - activation_('supersupersourcePV',_'supersourcePV0') + non_zero_pv >= 0
_C30: flow_('supersupersourcePV',_'supersourcePV14') <= 0
_C31: - activation_('supersupersourcePV',_'supersourcePV14') + non_zero_pv
 >= 0
_C32: flow_('supersupersourcePV',_'supersourcePV15') <= 0
_C33: - activation_('supersupersourcePV',_'supersourcePV15') + non_zero_pv
 >= 0
_C34: flow_('supersupersourcePV',_'supersourcePV16') <= 0
_C35: - activation_('supersupersourcePV',_'supersourcePV16') + non_zero_pv
 >= 0
_C36: flow_('supersupersourcePV',_'supersourcePV17') <= 0
_C37: - activation_('supersupersourcePV',_'supersourcePV17') + non_zero_pv
 >= 0
_C38: flow_('supersupersourcePV',_'supersourcePV18') <= 0
_C39: - activation_('supersupersourcePV',_'supersourcePV18') + non_zero_pv
 >= 0
_C4: flow_('supersupersourcePV',_'supersourcePV1') <= 0
_C40: flow_('supersupersourcePV',_'supersourcePV19') <= 0
_C41: - activation_('supersupersourcePV',_'supersourcePV19') + non_zero_pv
 >= 0
_C42: flow_('supersupersourcePV',_'supersourcePV20') <= 0
_C43: - activation_('supersupersourcePV',_'supersourcePV20') + non_zero_pv
 >= 0
_C44: flow_('supersupersourcePV',_'supersourcePV21') <= 0
_C45: - activation_('supersupersourcePV',_'supersourcePV21') + non_zero_pv
 >= 0
_C46: flow_('supersupersourcePV',_'supersourcePV22') <= 0
_C47: - activation_('supersupersourcePV',_'supersourcePV22') + non_zero_pv
 >= 0
_C48: flow_('supersupersourcePV',_'supersourcePV23') <= 0
_C49: - activation_('supersupersourcePV',_'supersourcePV23') + non_zero_pv
 >= 0
_C5: - activation_('supersupersourcePV',_'supersourcePV1') + non_zero_pv >= 0
_C50: flow_('supersupersourcePV',_'supersourcePV24') <= 0
_C51: - activation_('supersupersourcePV',_'supersourcePV24') + non_zero_pv
 >= 0
_C52: - flow_('supersourcePV0',_'PV0') - flow_('supersourcePV0',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV0') = 0
_C53: - flow_('supersourcePV1',_'PV0') - flow_('supersourcePV1',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV1') = 0
_C54: - flow_('supersourcePV2',_'PV0') - flow_('supersourcePV2',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV2') = 0
_C55: - flow_('supersourcePV3',_'PV0') - flow_('supersourcePV3',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV3') = 0
_C56: - flow_('supersourcePV4',_'PV0') - flow_('supersourcePV4',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV4') = 0
_C57: - flow_('supersourcePV5',_'PV0') - flow_('supersourcePV5',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV5') = 0
_C58: - flow_('supersourcePV6',_'PV0') - flow_('supersourcePV6',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV6') = 0
_C59: - flow_('supersourcePV7',_'PV0') - flow_('supersourcePV7',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV7') = 0
_C6: flow_('supersupersourcePV',_'supersourcePV2') <= 0
_C60: - flow_('supersourcePV8',_'PV0') - flow_('supersourcePV8',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV8') = 0
_C61: - flow_('supersourcePV9',_'PV0') - flow_('supersourcePV9',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV9') = 0
_C62: - flow_('supersourcePV10',_'PV0') - flow_('supersourcePV10',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV10') = 0
_C63: - flow_('supersourcePV11',_'PV0') - flow_('supersourcePV11',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV11') = 0
_C64: - flow_('supersourcePV12',_'PV0') - flow_('supersourcePV12',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV12') = 0
_C65: - flow_('supersourcePV13',_'PV0') - flow_('supersourcePV13',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV13') = 0
_C66: - flow_('supersourcePV14',_'PV0') - flow_('supersourcePV14',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV14') = 0
_C67: - flow_('supersourcePV15',_'PV0') - flow_('supersourcePV15',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV15') = 0
_C68: - flow_('supersourcePV16',_'PV0') - flow_('supersourcePV16',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV16') = 0
_C69: - flow_('supersourcePV17',_'PV0') - flow_('supersourcePV17',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV17') = 0
_C7: - activation_('supersupersourcePV',_'supersourcePV2') + non_zero_pv >= 0
_C70: - flow_('supersourcePV18',_'PV0') - flow_('supersourcePV18',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV18') = 0
_C71: - flow_('supersourcePV19',_'PV0') - flow_('supersourcePV19',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV19') = 0
_C72: - flow_('supersourcePV20',_'PV0') - flow_('supersourcePV20',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV20') = 0
_C73: - flow_('supersourcePV21',_'PV0') - flow_('supersourcePV21',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV21') = 0
_C74: - flow_('supersourcePV22',_'PV0') - flow_('supersourcePV22',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV22') = 0
_C75: - flow_('supersourcePV23',_'PV0') - flow_('supersourcePV23',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV23') = 0
_C76: - flow_('supersourcePV24',_'PV0') - flow_('supersourcePV24',_'PV1')
 + flow_('supersupersourcePV',_'supersourcePV24') = 0
_C77: - flow_('PV0',_'Battery0') - flow_('PV0',_'Consumption0')
 - flow_('PV0',_'supersink') + flow_('supersourcePV0',_'PV0')
 + flow_('supersourcePV1',_'PV0') + flow_('supersourcePV10',_'PV0')
 + flow_('supersourcePV11',_'PV0') + flow_('supersourcePV12',_'PV0')
 + flow_('supersourcePV13',_'PV0') + flow_('supersourcePV14',_'PV0')
 + flow_('supersourcePV15',_'PV0') + flow_('supersourcePV16',_'PV0')
 + flow_('supersourcePV17',_'PV0') + flow_('supersourcePV18',_'PV0')
 + flow_('supersourcePV19',_'PV0') + flow_('supersourcePV2',_'PV0')
 + flow_('supersourcePV20',_'PV0') + flow_('supersourcePV21',_'PV0')
 + flow_('supersourcePV22',_'PV0') + flow_('supersourcePV23',_'PV0')
 + flow_('supersourcePV24',_'PV0') + flow_('supersourcePV3',_'PV0')
 + flow_('supersourcePV4',_'PV0') + flow_('supersourcePV5',_'PV0')
 + flow_('supersourcePV6',_'PV0') + flow_('supersourcePV7',_'PV0')
 + flow_('supersourcePV8',_'PV0') + flow_('supersourcePV9',_'PV0') = 0
_C78: - flow_('PV1',_'Battery1') - flow_('PV1',_'Consumption1')
 - flow_('PV1',_'supersink') + flow_('supersourcePV0',_'PV1')
 + flow_('supersourcePV1',_'PV1') + flow_('supersourcePV10',_'PV1')
 + flow_('supersourcePV11',_'PV1') + flow_('supersourcePV12',_'PV1')
 + flow_('supersourcePV13',_'PV1') + flow_('supersourcePV14',_'PV1')
 + flow_('supersourcePV15',_'PV1') + flow_('supersourcePV16',_'PV1')
 + flow_('supersourcePV17',_'PV1') + flow_('supersourcePV18',_'PV1')
 + flow_('supersourcePV19',_'PV1') + flow_('supersourcePV2',_'PV1')
 + flow_('supersourcePV20',_'PV1') + flow_('supersourcePV21',_'PV1')
 + flow_('supersourcePV22',_'PV1') + flow_('supersourcePV23',_'PV1')
 + flow_('supersourcePV24',_'PV1') + flow_('supersourcePV3',_'PV1')
 + flow_('supersourcePV4',_'PV1') + flow_('supersourcePV5',_'PV1')
 + flow_('supersourcePV6',_'PV1') + flow_('supersourcePV7',_'PV1')
 + flow_('supersourcePV8',_'PV1') + flow_('supersourcePV9',_'PV1') = 0
_C79: - flow_('Battery0',_'Battery1') - flow_('Battery0',_'Consumption0')
 + flow_('PV0',_'Battery0') = 0
_C8: flow_('supersupersourcePV',_'supersourcePV3') <= 0
_C80: flow_('Battery0',_'Battery1') - flow_('Battery1',_'Consumption1')
 + flow_('PV1',_'Battery1') = 0
_C81: flow_('Battery0',_'Consumption0') - flow_('Consumption0',_'supersink')
 + flow_('PV0',_'Consumption0') + flow_('supersource',_'Consumption0') = 0
_C82: flow_('Battery1',_'Consumption1') - flow_('Consumption1',_'supersink')
 + flow_('PV1',_'Consumption1') + flow_('supersource',_'Consumption1') = 0
_C9: - activation_('supersupersourcePV',_'supersourcePV3') + non_zero_pv >= 0
battery_capacityBattery0: - cap_bat + flow_('Battery0',_'Battery1') <= 0
charging_ratePV0: - cap_bat + flow_('PV0',_'Battery0') <= 0
charging_ratePV1: - cap_bat + flow_('PV1',_'Battery1') <= 0
discharging_rateBattery0: - cap_bat + flow_('Battery0',_'Consumption0') <= 0
discharging_rateBattery1: - cap_bat + flow_('Battery1',_'Consumption1') <= 0
Bounds
cap_bat <= 10
flow_('Consumption0',_'supersink') <= 1.408
0.408 <= flow_('Consumption1',_'supersink') <= 1.408
flow_('PV0',_'Consumption0') <= 0.408
flow_('PV0',_'supersink') <= 10
flow_('PV1',_'Consumption1') <= 0.408
flow_('PV1',_'supersink') <= 10
flow_('supersource',_'Consumption0') <= 10
flow_('supersource',_'Consumption1') <= 10
flow_('supersourcePV0',_'PV0') = 0
flow_('supersourcePV0',_'PV1') = 0
flow_('supersourcePV1',_'PV0') = 0
flow_('supersourcePV1',_'PV1') = 0
flow_('supersourcePV10',_'PV0') = 0
flow_('supersourcePV10',_'PV1') = 0
flow_('supersourcePV11',_'PV0') = 0
flow_('supersourcePV11',_'PV1') = 0
flow_('supersourcePV12',_'PV0') = 0
flow_('supersourcePV12',_'PV1') = 0
flow_('supersourcePV13',_'PV0') = 0
flow_('supersourcePV13',_'PV1') = 0
flow_('supersourcePV14',_'PV0') = 0
flow_('supersourcePV14',_'PV1') = 0
flow_('supersourcePV15',_'PV0') = 0
flow_('supersourcePV15',_'PV1') = 0
flow_('supersourcePV16',_'PV0') = 0
flow_('supersourcePV16',_'PV1') = 0
flow_('supersourcePV17',_'PV0') = 0
flow_('supersourcePV17',_'PV1') = 0
flow_('supersourcePV18',_'PV0') = 0
flow_('supersourcePV18',_'PV1') = 0
flow_('supersourcePV19',_'PV0') = 0
flow_('supersourcePV19',_'PV1') = 0
flow_('supersourcePV2',_'PV0') = 0
flow_('supersourcePV2',_'PV1') = 0
flow_('supersourcePV20',_'PV0') = 0
flow_('supersourcePV20',_'PV1') = 0
flow_('supersourcePV21',_'PV0') = 0
flow_('supersourcePV21',_'PV1') = 0
flow_('supersourcePV22',_'PV0') = 0
flow_('supersourcePV22',_'PV1') = 0
flow_('supersourcePV23',_'PV0') = 0
flow_('supersourcePV23',_'PV1') = 0
flow_('supersourcePV24',_'PV0') = 0
flow_('supersourcePV24',_'PV1') = 0
flow_('supersourcePV3',_'PV0') = 0
flow_('supersourcePV3',_'PV1') = 0
flow_('supersourcePV4',_'PV0') = 0
flow_('supersourcePV4',_'PV1') = 0
flow_('supersourcePV5',_'PV0') = 0
flow_('supersourcePV5',_'PV1') = 0
flow_('supersourcePV6',_'PV0') = 0
flow_('supersourcePV6',_'PV1') = 0
flow_('supersourcePV7',_'PV0') = 0
flow_('supersourcePV7',_'PV1') = 0
flow_('supersourcePV8',_'PV0') = 0
flow_('supersourcePV8',_'PV1') = 0
flow_('supersourcePV9',_'PV0') = 0
flow_('supersourcePV9',_'PV1') = 0
Binaries
activation_('supersupersourcePV',_'supersourcePV0')
activation_('supersupersourcePV',_'supersourcePV1')
activation_('supersupersourcePV',_'supersourcePV10')
activation_('supersupersourcePV',_'supersourcePV11')
activation_('supersupersourcePV',_'supersourcePV12')
activation_('supersupersourcePV',_'supersourcePV13')
activation_('supersupersourcePV',_'supersourcePV14')
activation_('supersupersourcePV',_'supersourcePV15')
activation_('supersupersourcePV',_'supersourcePV16')
activation_('supersupersourcePV',_'supersourcePV17')
activation_('supersupersourcePV',_'supersourcePV18')
activation_('supersupersourcePV',_'supersourcePV19')
activation_('supersupersourcePV',_'supersourcePV2')
activation_('supersupersourcePV',_'supersourcePV20')
activation_('supersupersourcePV',_'supersourcePV21')
activation_('supersupersourcePV',_'supersourcePV22')
activation_('supersupersourcePV',_'supersourcePV23')
activation_('supersupersourcePV',_'supersourcePV24')
activation_('supersupersourcePV',_'supersourcePV3')
activation_('supersupersourcePV',_'supersourcePV4')
activation_('supersupersourcePV',_'supersourcePV5')
activation_('supersupersourcePV',_'supersourcePV6')
activation_('supersupersourcePV',_'supersourcePV7')
activation_('supersupersourcePV',_'supersourcePV8')
activation_('supersupersourcePV',_'supersourcePV9')
non_zero_pv
End
