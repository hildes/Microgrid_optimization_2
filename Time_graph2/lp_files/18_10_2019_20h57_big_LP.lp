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
_C1: - cap_bat + flow_('Battery9',_'Battery10') <= 0
_C10: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV0')
 + flow_('supersupersourcePV',_'supersourcePV0') <= 0
_C11: - activation_('supersupersourcePV',_'supersourcePV0') + non_zero_pv >= 0
_C12: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV1')
 + flow_('supersupersourcePV',_'supersourcePV1') <= 0
_C13: - activation_('supersupersourcePV',_'supersourcePV1') + non_zero_pv >= 0
_C14: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV2')
 + flow_('supersupersourcePV',_'supersourcePV2') <= 0
_C15: - activation_('supersupersourcePV',_'supersourcePV2') + non_zero_pv >= 0
_C16: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV3')
 + flow_('supersupersourcePV',_'supersourcePV3') <= 0
_C17: - activation_('supersupersourcePV',_'supersourcePV3') + non_zero_pv >= 0
_C18: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV4')
 + flow_('supersupersourcePV',_'supersourcePV4') <= 0
_C19: - activation_('supersupersourcePV',_'supersourcePV4') + non_zero_pv >= 0
_C2: - cap_bat + flow_('Battery10',_'Battery11') <= 0
_C20: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV5')
 + flow_('supersupersourcePV',_'supersourcePV5') <= 0
_C21: - activation_('supersupersourcePV',_'supersourcePV5') + non_zero_pv >= 0
_C22: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV6')
 + flow_('supersupersourcePV',_'supersourcePV6') <= 0
_C23: - activation_('supersupersourcePV',_'supersourcePV6') + non_zero_pv >= 0
_C24: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV7')
 + flow_('supersupersourcePV',_'supersourcePV7') <= 0
_C25: - activation_('supersupersourcePV',_'supersourcePV7') + non_zero_pv >= 0
_C26: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV8')
 + flow_('supersupersourcePV',_'supersourcePV8') <= 0
_C27: - activation_('supersupersourcePV',_'supersourcePV8') + non_zero_pv >= 0
_C28: - 0.426152315252 activation_('supersupersourcePV',_'supersourcePV9')
 + flow_('supersupersourcePV',_'supersourcePV9') <= 0
_C29: - activation_('supersupersourcePV',_'supersourcePV9') + non_zero_pv >= 0
_C3: - cap_bat + flow_('PV9',_'Battery9') <= 0
_C30: - flow_('supersourcePV0',_'PV10') - flow_('supersourcePV0',_'PV11')
 - flow_('supersourcePV0',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV0') = 0
_C31: - flow_('supersourcePV1',_'PV10') - flow_('supersourcePV1',_'PV11')
 - flow_('supersourcePV1',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV1') = 0
_C32: - flow_('supersourcePV2',_'PV10') - flow_('supersourcePV2',_'PV11')
 - flow_('supersourcePV2',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV2') = 0
_C33: - flow_('supersourcePV3',_'PV10') - flow_('supersourcePV3',_'PV11')
 - flow_('supersourcePV3',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV3') = 0
_C34: - flow_('supersourcePV4',_'PV10') - flow_('supersourcePV4',_'PV11')
 - flow_('supersourcePV4',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV4') = 0
_C35: - flow_('supersourcePV5',_'PV10') - flow_('supersourcePV5',_'PV11')
 - flow_('supersourcePV5',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV5') = 0
_C36: - flow_('supersourcePV6',_'PV10') - flow_('supersourcePV6',_'PV11')
 - flow_('supersourcePV6',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV6') = 0
_C37: - flow_('supersourcePV7',_'PV10') - flow_('supersourcePV7',_'PV11')
 - flow_('supersourcePV7',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV7') = 0
_C38: - flow_('supersourcePV8',_'PV10') - flow_('supersourcePV8',_'PV11')
 - flow_('supersourcePV8',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV8') = 0
_C39: - flow_('supersourcePV9',_'PV10') - flow_('supersourcePV9',_'PV11')
 - flow_('supersourcePV9',_'PV9')
 + flow_('supersupersourcePV',_'supersourcePV9') = 0
_C4: - cap_bat + flow_('PV10',_'Battery10') <= 0
_C40: - flow_('Battery9',_'Battery10') - flow_('Battery9',_'Consumption9')
 + flow_('PV9',_'Battery9') = 0
_C41: - flow_('Battery10',_'Battery11') - flow_('Battery10',_'Consumption10')
 + flow_('Battery9',_'Battery10') + flow_('PV10',_'Battery10') = 0
_C42: flow_('Battery10',_'Battery11') - flow_('Battery11',_'Consumption11')
 + flow_('PV11',_'Battery11') = 0
_C5: - cap_bat + flow_('PV11',_'Battery11') <= 0
_C6: - cap_bat + flow_('Battery9',_'Consumption9') <= 0
_C7: - cap_bat + flow_('Battery10',_'Consumption10') <= 0
_C8: - cap_bat + flow_('Battery11',_'Consumption11') <= 0
_C9: 0.426152315252 activation_('supersupersourcePV',_'supersourcePV0')
 - flow_('supersupersourcePV',_'supersourcePV0') <= 0
Bounds
cap_bat <= 10
flow_('PV10',_'supersink') = 10
flow_('PV11',_'supersink') = 10
flow_('PV9',_'supersink') = 10
flow_('supersource',_'Consumption10') = 100
flow_('supersource',_'Consumption11') = 100
flow_('supersource',_'Consumption9') = 100
flow_('supersourcePV0',_'PV10') = 0.129477357356
flow_('supersourcePV0',_'PV11') = 0.219984721858
flow_('supersourcePV0',_'PV9') = 0.0766902360387
flow_('supersourcePV1',_'PV10') = 0.129477357356
flow_('supersourcePV1',_'PV11') = 0.219984721858
flow_('supersourcePV1',_'PV9') = 0.0766902360387
flow_('supersourcePV2',_'PV10') = 0.129477357356
flow_('supersourcePV2',_'PV11') = 0.219984721858
flow_('supersourcePV2',_'PV9') = 0.0766902360387
flow_('supersourcePV3',_'PV10') = 0.129477357356
flow_('supersourcePV3',_'PV11') = 0.219984721858
flow_('supersourcePV3',_'PV9') = 0.0766902360387
flow_('supersourcePV4',_'PV10') = 0.129477357356
flow_('supersourcePV4',_'PV11') = 0.219984721858
flow_('supersourcePV4',_'PV9') = 0.0766902360387
flow_('supersourcePV5',_'PV10') = 0.129477357356
flow_('supersourcePV5',_'PV11') = 0.219984721858
flow_('supersourcePV5',_'PV9') = 0.0766902360387
flow_('supersourcePV6',_'PV10') = 0.129477357356
flow_('supersourcePV6',_'PV11') = 0.219984721858
flow_('supersourcePV6',_'PV9') = 0.0766902360387
flow_('supersourcePV7',_'PV10') = 0.129477357356
flow_('supersourcePV7',_'PV11') = 0.219984721858
flow_('supersourcePV7',_'PV9') = 0.0766902360387
flow_('supersourcePV8',_'PV10') = 0.129477357356
flow_('supersourcePV8',_'PV11') = 0.219984721858
flow_('supersourcePV8',_'PV9') = 0.0766902360387
flow_('supersourcePV9',_'PV10') = 0.129477357356
flow_('supersourcePV9',_'PV11') = 0.219984721858
flow_('supersourcePV9',_'PV9') = 0.0766902360387
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
