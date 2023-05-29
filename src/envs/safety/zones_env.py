import numpy as np
import enum
import gym

from safety_gym.envs.engine import Engine

class zone(enum.Enum):
    JetBlack = 0
    White    = 1
    Blue     = 2
    Green    = 3
    Red      = 4
    Yellow   = 5
    Cyan     = 6
    Magenta  = 7

    def __lt__(self, sth):
        return self.value < sth.value

    def __str__(self):
        return self.name[0]

    def __repr__(self):
        return self.name

GROUP_ZONE = 7

class ZonesEnv(Engine):
    """
    This environment is a modification of the Safety-Gym's environment.
    There is no "goal circle" but rather a collection of zones that the
    agent has to visit or to avoid in order to finish the task.

    For now we only support the 'point' robot.
    """
    def __init__(self, zones:list, use_fixed_map:float, timeout:int, config=dict):
        walled = True
        self.DEFAULT.update({
            'observe_zones': False,
            'zones_num': 0,  # Number of hazards in an environment
            'zones_placements': None,  # Placements list for hazards (defaults to full extents)
            'zones_locations': [],  # Fixed locations to override placements
            'zones_keepout': 0.55,  # Radius of hazard keepout for placement
            'zones_size': 0.25,  # Radius of hazards
        })

        if (walled):
            world_extent = 2.5
            walls = [(i/10, j) for i in range(int(-world_extent * 10),int(world_extent * 10 + 1),1) for j in [-world_extent, world_extent]]
            walls += [(i, j/10) for i in [-world_extent, world_extent] for j in range(int(-world_extent * 10), int(world_extent * 10 + 1),1)]
            self.DEFAULT.update({
                'placements_extents': [-world_extent, -world_extent, world_extent, world_extent],
                'walls_num': len(walls),  # Number of walls
                'walls_locations': walls,  # This should be used and length == walls_num
                'walls_size': 0.1,  # Should be fixed at fundamental size of the world
            })

        self.zones = zones
        self.zone_types = list(set(zones))
        self.zone_types.sort()
        self.use_fixed_map = use_fixed_map
        self._rgb = {
            zone.JetBlack: [0, 0, 0, 1],
            zone.Blue    : [0, 0, 1, 1],
            zone.Green   : [0, 1, 0, 1],
            zone.Cyan    : [0, 1, 1, 1],
            zone.Red     : [1, 0, 0, 1],
            zone.Magenta : [1, 0, 1, 1],
            zone.Yellow  : [1, 1, 0, 1],
            zone.White   : [1, 1, 1, 1]
        }
        self.zone_rgbs = np.array([self._rgb[haz] for haz in self.zones])

        # self.zones_position = [
        #     # np.array([-1., 0., 0.02]), np.array([-0.5, -1.5, 0.02]), np.array([1., -0.5, 0.02]), np.array([2, 2, 0.02]),  # W
        #     np.array([-1., 0., 0.02]), np.array([-0.5, -1.5, 0.02]), np.array([1., -0.5, 0.02]),
        #     np.array([-0.5, 1., 0.02]), np.array([1., 0.5, 0.02]),  # Y
        #     np.array([-1.75, 1., 0.02]), np.array([-1.5, -1., 0.02]), np.array([1.5, -1.5, 0.02]),
        #     np.array([2., -0.5, 0.02]),  # J
        #     np.array([-2., 0., 0.02]), np.array([0.5, -1.5, 0.02])]  # RY
        #
        # self.zones_position = [
        #     np.array([-0.5, 0.5, 0.02]), np.array([0.75, 0., 0.02]),  # J
        #     np.array([-0.75, -0.5, 0.02]), np.array([0.5, -0.75, 0.02]), np.array([-1.5, 0.5, 0.02]),  # R
        #     np.array([0.5, 1.5, 0.02]), np.array([-1., 2., 0.02]),  # W
        #     np.array([-1.25, 1.25, 0.02]), np.array([0., 1., 0.02]), np.array([1., 1., 0.02]),  # Y
        # ]

        # Position setting for Zones-25-v1
        self.zones_position = [
            np.array([0., 0., 0.02]), np.array([1.0, 0.5, 0.02]), np.array([-1.0, 2.0, 0.02]),  # J
            np.array([-0.5, -1.0, 0.02]), np.array([0.75, -0.5, 0.02]),  # R
            np.array([-1.5, -1.0, 0.02]), np.array([-1., 0.5, 0.02]), np.array([0.5, 1.0, 0.02]),  # W
            np.array([-0.5, 1., 0.02]), np.array([-0.5, -2.0, 0.02]), np.array([1.0, 2.0, 0.02]),  # Y
        ]

        parent_config = {
            'robot_base': 'xmls/point.xml',
            'task': 'none',
            'lidar_num_bins': 16,
            'observe_zones': True,
            'zones_num': len(zones),
            'num_steps': timeout,

            # 'placements_extents': [-4, -4, 4, 4],
            # 'observe_vases':True,
            # 'vases_num': 1,
            # 'observe_pillars': True,
            # 'pillars_num': 1,
            # 'observe_gremlins': True,
            # 'gremlins_num': 1,
        }
        parent_config.update(config)

        super().__init__(parent_config)

    @property
    def zones_pos(self):
        ''' Helper to get the zones positions from layout '''
        return [self.data.get_body_xpos(f'zone{i}').copy() for i in range(self.zones_num)]

    def build_observation_space(self):
        super().build_observation_space()

        if self.observe_zones:
            for zone_type in self.zone_types:
                self.obs_space_dict.update({f'zones_lidar_{zone_type}': gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)})

        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(self.obs_space_dict)

    def build_placements_dict(self):
        super().build_placements_dict()

        if self.zones_num: #self.constrain_hazards:
            self.placements.update(self.placements_dict_from_object('zone'))

    def build_world_config(self):
        world_config = super().build_world_config()

        # set default env
        world_config['robot_xy'] = np.array([0., -1.5])
        world_config['robot_rot'] = float(2.)
        # world_config['robot_xy'] = np.array([0., -1.5])
        # world_config['robot_rot'] = float(1.6)

        for i in range(self.zones_num):
            name = f'zone{i}'
            geom = {'name': name,
                    'size': [self.zones_size, 1e-2],#self.zones_size / 2],
                    # 'pos': np.r_[self.layout[name], 2e-2],#self.zones_size / 2 + 1e-2],
                    'pos': self.zones_position[i],  # self.zones_size / 2 + 1e-2],
                    'rot': self.random_rot(),
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_ZONE,
                    'rgba': self.zone_rgbs[i] * [1, 1, 1, 0.25]} #0.1]}  # transparent
            world_config['geoms'][name] = geom

        return world_config

    def build_obs(self):
        obs = super().build_obs()

        if self.observe_zones:
            for zone_type in self.zone_types:
                ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                pos_in_type = list(np.array(self.zones_pos)[ind])

                obs[f'zones_lidar_{zone_type}'] = self.obs_lidar(pos_in_type, GROUP_ZONE)

        return obs


    def render_lidars(self):
        offset = super().render_lidars()

        if self.render_lidar_markers:
            for zone_type in self.zone_types:
                if f'zones_lidar_{zone_type}' in self.obs_space_dict:
                    ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                    pos_in_type = list(np.array(self.zones_pos)[ind])

                    self.render_lidar(pos_in_type, np.array([self._rgb[zone_type]]), offset, GROUP_ZONE)
                    offset += self.render_lidar_offset_delta

        return offset

    def seed(self, seed=None):
        if (self.use_fixed_map): self._seed = seed


class LTLZonesEnv(ZonesEnv):
    def __init__(self, zones:list, use_fixed_map:float, timeout:int, config={}):
        super().__init__(zones=zones, use_fixed_map=use_fixed_map, timeout=timeout, config=config)

    def get_propositions(self):
        return [str(i) for i in self.zone_types]

    def get_events(self):
        events = ""
        for h_inedx, h_pos in enumerate(self.zones_pos):
            h_dist = self.dist_xy(h_pos)
            if h_dist <= self.zones_size:
                # We assume the agent to be in one zone at a time
                events += str(self.zones[h_inedx])

        return events

class ZonesEnv1(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.Red], use_fixed_map=False, timeout=1000)

class ZonesEnv1Fixed(LTLZonesEnv):
    def __init__(self):
        config = {
            # 'placements_extents': [-1.5, -1.5, 1.5, 1.5]
        }
        super().__init__(zones=[zone.Red], use_fixed_map=True, timeout=1000, config=config)

class ZonesEnv5(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.Red, zone.Red, zone.White, zone.White,  zone.Yellow, zone.Yellow], use_fixed_map=False, timeout=1000)

class ZonesEnv5Fixed(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.Red, zone.Red, zone.White, zone.White,  zone.Yellow, zone.Yellow], use_fixed_map=True, timeout=1000)

class ZonesEnv5PROFixed(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.JetBlack,
                                zone.Red, zone.Red, zone.Red,
                                zone.White, zone.White, zone.White,
                                zone.Yellow, zone.Yellow, zone.Yellow],
                         use_fixed_map=True, timeout=1000)


class ZonesEnv6Fixed(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.JetBlack, zone.JetBlack,
                                zone.Red,
                                zone.White, zone.White, zone.White, zone.White,
                                zone.Yellow],
                         use_fixed_map=True, timeout=1000)

class ZonesEnv7Fixed(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.JetBlack, zone.JetBlack,
                                zone.Red, zone.Red, zone.Red,
                                zone.White, zone.White,
                                zone.Yellow,  zone.Yellow,  zone.Yellow,
                                ],
                         use_fixed_map=True, timeout=1500)

class ZonesEnv25Fixed(LTLZonesEnv):
    def __init__(self):
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.JetBlack,
                                zone.Red, zone.Red,
                                zone.White, zone.White, zone.White,
                                zone.Yellow,  zone.Yellow,  zone.Yellow,
                                ],
                         use_fixed_map=True, timeout=1000)

if __name__ == '__main__':
    import numpy as np
    env = ZonesEnv5Fixed()
    print(env.observation_space)
    s = env.reset()
    print(np.shape(s))
