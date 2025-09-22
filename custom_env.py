# fichier: custom_env.py
import os
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv


from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv, DEFAULT_CAMERA_CONFIG


from gymnasium.utils import EzPickle


class HumanoidObstaclesEnv(HumanoidEnv):
    """
    Version finale de l'environnement personnalisé.
    Définit l'espace d'observation manuellement pour une initialisation correcte.
    """

    def __init__(self, **kwargs):
        # On force l'utilisation de notre fichier XML personnalisé
        xml_path = os.path.join(os.path.dirname(__file__), "humanoid_with_obstacles.xml")

        # ================================================================= #
        # == COPIE EXACTE DE L'INITIALISATION DE HUMANOID-V5 COMMENCE ICI == #
        # ================================================================= #

        # On définit toutes les variables de la même manière que l'original
        self._forward_reward_weight = 1.25
        self._ctrl_cost_weight = 0.1
        self._contact_cost_weight = 5e-7
        self._contact_cost_range = (-np.inf, 10.0)
        self._healthy_reward = 5.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (1.0, 2.0)
        self._reset_noise_scale = 1e-2
        self._exclude_current_positions_from_observation = True

        self._include_cinert_in_observation = True
        self._include_cvel_in_observation = True
        self._include_qfrc_actuator_in_observation = True
        self._include_cfrc_ext_in_observation = True

        # Appel au constructeur du "grand-parent" avec notre fichier XML et la config de caméra
        MujocoEnv.__init__(
            self,
            xml_path,  # 👈 NOTRE MODIFICATION PRINCIPALE
            frame_skip=5,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,  # 👈 2. On utilise la constante importée
            **kwargs,
        )

        # Copie de la logique pour calculer dynamiquement la taille de l'observation
        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * self._exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * self._include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * self._include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * self._include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * self._include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        EzPickle.__init__(self, **kwargs)

    # La méthode step() pour la pénalité de collision reste la même
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        collision_penalty = 0
        obstacle_names = ["hurdle", "pole"]

        for contact in self.data.contact:
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name

            is_robot_part = geom1_name is not None and (
                        "torso" in geom1_name or "thigh" in geom1_name or "shin" in geom1_name)
            is_obstacle = geom2_name in obstacle_names

            if is_robot_part and is_obstacle:
                collision_penalty -= 2
                break

        reward += collision_penalty

        return observation, reward, terminated, truncated, info


# L'enregistrement de l'environnement ne change pas
gym.register(
    id='HumanoidObstacles-v0',
    entry_point='custom_env:HumanoidObstaclesEnv',
    max_episode_steps=1000,
)