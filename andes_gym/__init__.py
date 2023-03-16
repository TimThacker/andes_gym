from gym.envs.registration import register
from andes_gym.envs.andes_freq import AndesFreqControl
from andes_gym.envs.andes_prim_freq import AndesPrimaryFreqControl
from andes_gym.envs.AndesPrimFreqTest import AndesPrimaryFreqControlTest
from andes_gym.envs.wecc_andes_prim_freq import AndesPrimaryFreqControlWECC

register(
    id='AndesFreqControl-v0',
    entry_point='andes_gym:AndesFreqControl',
)

register(
    id='AndesPrimaryFreqControl-v0',
    entry_point='andes_gym:AndesPrimaryFreqControl',
)

register(
    id='AndesPrimaryFreqControlWECC-v0',
    entry_point='andes_gym:AndesPrimaryFreqControlWECC',
)
register(
    id='AndesPrimaryFreqControlTest-v0',
    entry_point='andes_gym:AndesPrimaryFreqControlTest',
)
