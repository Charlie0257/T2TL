from gym.envs.registration import register

from envs.safety.zones_env import ZonesEnv

__all__ = ["ZonesEnv"]


### Safety Envs

register(
    id='Zones-25-v1',
    entry_point='envs.safety.zones_env:ZonesEnv25Fixed')