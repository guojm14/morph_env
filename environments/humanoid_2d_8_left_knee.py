import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

import os

class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate(
            (
                position,
                velocity
            )
        )

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        #alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        #reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        qpos = self.sim.data.qpos
        done = False
        return self._get_obs(), reward, done, {}

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20