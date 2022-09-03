from gym.envs.registration import register
from andes_gym.envs.andes_freq import AndesFreqControl
from andes_gym.envs.andes_freq import AndesPrimaryFreqControl

register(
    id='AndesFreqControl-v0',
    entry_point='andes_gym:AndesFreqControl',
)

register(
    id='AndesPrimaryFreqControl-v0',
    entry_point='andes_gym:AndesPrimaryFreqControl',
)
