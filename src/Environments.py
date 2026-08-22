import gym
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

class MultiAgentEnv(gym.Env):

    def __init__(self, num_agents: int, grid_size: float, max_vel: float = 0.1, max_omega: float = 0.01,  max_moment: float = 0.1, max_force: float = 1, num_obstacles: int = 2, use_quaternion: bool = False, Point_mass: bool = False):
        super(MultiAgentEnv, self).__init__()

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.dt = 0.01
        self.max_vel = max_vel
        self.max_omega = max_omega
        self.max_force = max_force
        self.max_moment = max_moment
        self.num_obstacles = num_obstacles
        self.obs_shape = 12 + (num_agents - 1) * 6 + 3 * num_obstacles + 3
        self.neighbor_radius = 2

        self.mass = 1
        self.a = 0.1
        self.arm = 0.1

        self.I = 1/6 * self.mass * self.a **2 * np.diag([1, 1, 1])
        self.I_inv = np.linalg.inv(self.I)


        self.d_close = self.a

        self.alpha_pos = 1  
        self.alpha_rcol = 5
        self.alpha_force = 0.1
        self.alpha_attitude = 0.1



        self.use_quaternion = use_quaternion
        self.point_mass = Point_mass
        self.min_reward = self._calculate_min_reward()
        self.reset()

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        self._initialize_obstacles()
        self._initialize_states()
        self._initialize_destinations()

        self.collision_count = 0
        self.success_count = 0

        return self._get_obs()

    def _initialize_obstacles(self) -> np.ndarray:
        """
        initializes obstacle positions.
        """
        self.obstacles = np.array([[0.1, 0.6, 0.4], [0.9, 0.4, 0.4]])

    def _initialize_states(self) -> np.ndarray:
        """
        Initializes the states of the agents.
        """
        self.eul_ang  = np.zeros((self.num_agents, 3))
        self.quat = R.from_euler('xyz', self.eul_ang).as_quat()
        self.omega = np.zeros((self.num_agents, 3))
        # positions = []
        # for _ in range(self.num_agents):
        #     while True:
        #         cand = np.random.uniform(0 + 0.1, self.grid_size - 0.1, (3,))
        #         if  all(np.linalg.norm(cand - p2) >= self.a for p2 in self.obstacles):
        #             positions.append(cand)
        #             break
        # self.pos = np.asarray(positions)
        self.pos = np.array([[0.15, 0.1, 0.5], [0.85, 0.1, 0.5]])
        self.vel = np.zeros((self.num_agents, 3))

    def _initialize_destinations(self) -> np.ndarray:
        """
        Initializes the destinations for the agents.
        """
        # destinations = []
        # for _ in range(self.num_agents):
        #     while True:
        #         cand = np.random.uniform(self.a, self.grid_size - self.a, 3)
        #         if all(np.linalg.norm(cand - p1) >= self.a for p1 in self.pos) and all(np.linalg.norm(cand - p2) >= self.a for p2 in self.obstacles):
        #             destinations.append(cand)
        #             break
        # self.destinations = np.asarray(destinations)

        self.destinations = np.array([[0.4, 0.8, 0.4], [0.8, 0.8, 0.6]])
        return self.destinations

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Executes one time step within the environment.
        """

        if(self.point_mass):

            forces = action
        else:
            forces, torques = action[:,:3], action[:,3:]


            self.euler_equations(torques)
            self.kinematic_attitude()
            self.kinetic_trans(forces)

            self.pos = np.clip(self.pos, 0, self.grid_size)
            self.vel = np.clip(self.vel, -self.max_vel, self.max_vel)
            self.omega = np.clip(self.omega, -self.max_omega, self.max_omega)

        obs = self._get_obs()
        reward, reward_components = self._compute_reward(action)
        done = self._check_done()

        return obs, reward, reward_components, done

    def _get_obs(self) -> np.ndarray:
        """
        Generates observations for all agents.
        """
        observations = []
        for i in range(self.num_agents):
            agent_obs = self._get_agent_obs(i)
            observations.append(agent_obs)
        return np.array(observations)

    def _get_agent_obs(self, agent_id: int) -> np.ndarray:
        """
        Generates observation for a single agent.
        """
        self_obs = np.hstack([
            self.pos[agent_id],
            self.vel[agent_id],
            self.eul_ang[agent_id],
            self.omega[agent_id],
            self.destinations[agent_id]
        ])

        neighbors = self.get_neighbors(agent_id)
        neighbors_obs = []
        for i in neighbors:
            relative_position = (self.pos[agent_id] - self.pos[i])
            relative_velocity = (self.vel[agent_id] - self.vel[i])
            neighbors_obs.append(np.hstack((relative_position, relative_velocity)))

        while len(neighbors_obs) < self.num_agents - 1:
            neighbors_obs.append(np.zeros((6,)))
        if len(neighbors_obs) >= 1:
          neighbors_obs = np.concatenate(neighbors_obs, axis=0)

        obstacles_obs = []
        for obstacle in self.obstacles:
            distance_to_obstacle = (obstacle - self.pos[agent_id])
            obstacles_obs.append(distance_to_obstacle)

        obstacles_obs = np.array(obstacles_obs)
        while len(obstacles_obs) < self.num_obstacles:
            obstacles_obs = np.append(obstacles_obs, np.zeros(3))

        obstacles_obs = np.concatenate(obstacles_obs, axis=0)
        obs = np.concatenate((self_obs, neighbors_obs, obstacles_obs))
        return obs

    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        Gets the neighbors of a given agent.
        """
        neighbors = []
        for i in range(self.num_agents):
            if i != agent_id:
                distance = np.linalg.norm(self.pos[agent_id] - self.pos[i])
                if distance < self.neighbor_radius:
                    neighbors.append(i)
        return neighbors

    def _compute_reward(self, actions: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Computes the reward for all agents.
        """
        reward = np.zeros(self.num_agents)
        reward_components = {
            'distance_to_destination': np.zeros(self.num_agents),
            'collision_with_obstacles': np.zeros(self.num_agents),
            'collision_with_agents': np.zeros(self.num_agents),
        }

        for i in range(self.num_agents):
            dist_to_dest = np.linalg.norm(self.pos[i] - self.destinations[i])
            r_pos = -self.alpha_pos * (dist_to_dest ** 2)
            reward_components['distance_to_destination'][i] = r_pos

            r_col = 0
            r_col += self._compute_collision_penalty(i)
            reward_components['collision_with_obstacles'][i] = r_col

            for j in range(self.num_agents):
                if i != j:
                    dist_to_agent = np.linalg.norm(self.pos[i] - self.pos[j])
                    if dist_to_agent < self.d_close:
                        reward_components['collision_with_agents'][i] -= self.alpha_rcol * (1 - (dist_to_agent / self.d_close))
                        r_col += reward_components['collision_with_agents'][i]
                        self.collision_count += 1

            reward[i] = r_pos + r_col
        return reward, reward_components

    def _compute_collision_penalty(self, agent_id: int) -> float:
        """
        Computes the penalty for colliding with obstacles.
        """
        r_col = 0.0
        for obstacle in self.obstacles:
            dist_xy = np.linalg.norm(self.pos[agent_id][:2] - obstacle[:2])
            if dist_xy < self.d_close:
                r_col -= self.alpha_rcol * (1.0 - dist_xy / self.d_close)
                self.collision_count += 1
        return r_col

    def _calculate_min_reward(self) -> float:
        """
        Calculates the minimum possible reward.
        """
        max_distance = - np.sqrt(3) * self.grid_size
        max_penalty_pos = - self.alpha_pos * (max_distance ** 2)
        max_penalty_col = - self.alpha_rcol * self.num_obstacles
        max_penalty_agent_col = - self.alpha_rcol * (self.num_agents - 1)
        min_reward = max_penalty_pos + max_penalty_agent_col + max_penalty_col
        return min_reward

    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalizes the rewards.
        """
        return rewards / self.min_reward

    def _check_done(self) -> bool:
        """
        Checks if the episode is done for each agent.
        """
        done_list = []
        for i in range(self.num_agents):
            pos_condition = np.linalg.norm(self.pos[i] - self.destinations[i]) < self.a
            done_list.append(pos_condition)
        if all(done_list):
            self.success_count += 1
        return done_list

    def euler_equations(self, torques: np.ndarray):
        """
        Updates the angular velocity of the agents.
        """
        for i in range(self.num_agents):
            d_omega_dt = self.I_inv @ (torques[i] - self.anti_symmtreic(self.omega[i]) @ self.I @ self.omega[i])
            self.omega[i] += self.dt * d_omega_dt

    def kinematic_attitude(self):
        """
        Updates the attitude of the agents.
        """
        for i in range(self.num_agents):
            if self.use_quaternion:
                omega_body = self.omega[i]
                quat = self.quaternions[i]
                omega_quat = np.hstack(([0.0], omega_body))
                quat_dot = 0.5 * self.quat_multiply(quat, omega_quat)
                quat_new = quat + self.dt * quat_dot
                quat_new /= np.linalg.norm(quat_new)
                self.quaternions[i] = quat_new
            else:
                phi, theta, psi = self.eul_ang[i]
                T = np.array([
                    [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, (np.cos(theta) + 10**(-6)), np.cos(phi) / (np.cos(theta) + 10**(-6))]
                ])
                eul_rates = T @ self.omega[i]
                self.eul_ang[i] += self.dt * eul_rates
                self.eul_ang[i] = np.arctan2(np.sin(self.eul_ang[i]), np.cos(self.eul_ang[i]))

    @staticmethod
    def quat_multiply(q, r):
        return np.array([
            q[3]*r[0] + q[0]*r[3] + q[1]*r[2] - q[2]*r[1],
            q[3]*r[1] - q[0]*r[2] + q[1]*r[3] + q[2]*r[0],
            q[3]*r[2] + q[0]*r[1] - q[1]*r[0] + q[2]*r[3],
            q[3]*r[3] - q[0]*r[0] - q[1]*r[1] - q[2]*r[2]
        ])

    def kinetic_trans(self, forces):
        """
        Updates the translational motion of the agents.
        """
        for i in range(self.num_agents):
            omega_cross = self.anti_symmtreic(self.omega[i])
            force_term = forces[i] / self.mass
            coriolis_term = omega_cross @ self.vel[i]
            acceleration = force_term - coriolis_term
            self.vel[i] += self.dt * acceleration

            if self.use_quaternion:
                quat = self.quaternions[i]
                R_mat = R.from_quat(quat).as_matrix()
            else:
                phi, theta, psi = self.eul_ang[i]
                R_x = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
                R_y = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
                R_z = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
                R_mat = R_z @ R_y @ R_x

            self.pos[i] += self.dt * (R_mat.T @ self.vel[i])

    def anti_symmtreic(self, v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], 0, 0]])
