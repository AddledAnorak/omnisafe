# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of a basic algorithm framework for offline algorithms."""

from __future__ import annotations

import time
from abc import abstractmethod

import torch

import dsrl
import gymnasium as gym
from tqdm import tqdm

from omnisafe.adapter import OfflineAdapter
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.logger import Logger
from omnisafe.common.offline.dataset import OfflineDataset
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config


# changes have been made by shoan raj (@AddledAnorak)
class DSRLDataset:
    def __init__(self, 
        env_id: str = "OfflinePointGoal2Gymnasium-v0",
        batch_size: int = 256,
        **kwargs
    ):
        env = gym.make(env_id)
        self.batch_size = batch_size
        dataset = env.get_dataset()
        self.obs = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.actions = torch.tensor(dataset['actions'], dtype=torch.float32)
        self.costs = torch.tensor(dataset['costs'], dtype=torch.float32)
        self.rewards = torch.tensor(dataset['rewards'], dtype=torch.float32)
        self.next_obs = torch.tensor(dataset['next_observations'], dtype=torch.float32)
        self.dones = torch.tensor(dataset['terminals'] + dataset['timeouts'], dtype=torch.float32)

    
    def sample(self):
        indices = torch.randint(low=0, high=(self.obs.shape[0]), size=(self.batch_size,), dtype=torch.int64)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.costs[indices],
            self.next_obs[indices],
            self.dones[indices]
        )



class BaseOffline(BaseAlgo):
    """Base class for offline algorithms."""

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize an instance of :class:`BaseOffline`."""
        super().__init__(env_id, cfgs)

        self._actor: Actor
        self.epoch: int = 0

    def _init(self) -> None:
        print(f'you chose: {self._cfgs.train_cfgs.dataset}')
        # self._dataset = OfflineDataset(
        self._dataset = DSRLDataset(
            # self._cfgs.train_cfgs.dataset,
            env_id=self._cfgs.train_cfgs.dataset,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            # device=self._device,
        )

    def _init_env(self) -> None:
        self._env = OfflineAdapter(self._env_id, self._seed, self._cfgs)

    def _init_log(self) -> None:
        """Log info each epoch.

        +----------------+--------------------------------+
        | Things to log  | Description                    |
        +================+================================+
        | Metrics/EpCost | Average cost of the epoch.     |
        +-------------------------------------------------+
        | Metrics/EpRet  | Average return of the epoch.   |
        +-------------------------------------------------+
        | Metrics/EpLen  | Average length of the epoch.   |
        +-------------------------------------------------+
        | Time/Total     | Total time.                    |
        +-------------------------------------------------+
        | Time/Epoch     | Time in each epoch.            |
        +-------------------------------------------------+
        | Time/Update    | Update time in each epoch.     |
        +-------------------------------------------------+
        | Time/Evaluate  | Evaluate time in each epoch.   |
        +-------------------------------------------------+
        | Train/Epoch    | Current epoch.                 |
        +-------------------------------------------------+
        | TotalSteps     | Total steps of the experiment. |
        +-------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name + f'-{self._cfgs.train_cfgs.dataset}',
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        self._logger.register_key('Metrics/EpRet')
        self._logger.register_key('Metrics/EpCost')
        self._logger.register_key('Metrics/EpLen')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('TotalSteps')

    def learn(self) -> tuple[float, float, float]:
        """Learn the policy."""
        self._logger.log('Start training ...')

        start_time = time.time()
        epoch_time = time.time()

        for step in tqdm(range(self._cfgs.train_cfgs.total_steps)):
            batch = self._dataset.sample()
            self._train(batch)

            if (step + 1) % self._cfgs.algo_cfgs.steps_per_epoch == 0:
                self.epoch = (step + 1) // self._cfgs.algo_cfgs.steps_per_epoch
                self._logger.store(**{'Time/Update': time.time() - epoch_time})
                eval_time = time.time()
                # self._evaluate()

                self._logger.store(
                    {
                        'Time/Evaluate': time.time() - eval_time,
                        'Time/Epoch': time.time() - epoch_time,
                        'Time/Total': time.time() - start_time,
                        'Train/Epoch': self.epoch,
                        'TotalSteps': step + 1,
                    },
                )

                epoch_time = time.time()
                self._logger.dump_tabular()

                # save model to disk
                if self.epoch % self._cfgs.logger_cfgs.save_model_freq == 0:
                    self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    @abstractmethod
    def _train(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> None:
        """Train the model."""

    def _evaluate(self) -> None:
        """Evaluate the model."""
        self._env.evaluate(
            evaluate_epoisodes=self._cfgs.train_cfgs.evaluate_epoisodes,
            logger=self._logger,
            agent=self._actor,
        )



# backup original class after shoan's changes
class BaseOfflineOriginal(BaseAlgo):
    """Base class for offline algorithms."""

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize an instance of :class:`BaseOffline`."""
        super().__init__(env_id, cfgs)

        self._actor: Actor
        self.epoch: int = 0

    def _init(self) -> None:
        self._dataset = OfflineDataset(
            self._cfgs.train_cfgs.dataset,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            device=self._device,
        )

    def _init_env(self) -> None:
        self._env = OfflineAdapter(self._env_id, self._seed, self._cfgs)

    def _init_log(self) -> None:
        """Log info each epoch.

        +----------------+--------------------------------+
        | Things to log  | Description                    |
        +================+================================+
        | Metrics/EpCost | Average cost of the epoch.     |
        +-------------------------------------------------+
        | Metrics/EpRet  | Average return of the epoch.   |
        +-------------------------------------------------+
        | Metrics/EpLen  | Average length of the epoch.   |
        +-------------------------------------------------+
        | Time/Total     | Total time.                    |
        +-------------------------------------------------+
        | Time/Epoch     | Time in each epoch.            |
        +-------------------------------------------------+
        | Time/Update    | Update time in each epoch.     |
        +-------------------------------------------------+
        | Time/Evaluate  | Evaluate time in each epoch.   |
        +-------------------------------------------------+
        | Train/Epoch    | Current epoch.                 |
        +-------------------------------------------------+
        | TotalSteps     | Total steps of the experiment. |
        +-------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name + f'-{self._cfgs.train_cfgs.dataset}',
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        self._logger.register_key('Metrics/EpRet')
        self._logger.register_key('Metrics/EpCost')
        self._logger.register_key('Metrics/EpLen')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('TotalSteps')

    def learn(self) -> tuple[float, float, float]:
        """Learn the policy."""
        self._logger.log('Start training ...')

        start_time = time.time()
        epoch_time = time.time()

        for step in range(self._cfgs.train_cfgs.total_steps):
            batch = self._dataset.sample()
            self._train(batch)

            if (step + 1) % self._cfgs.algo_cfgs.steps_per_epoch == 0:
                self.epoch = (step + 1) // self._cfgs.algo_cfgs.steps_per_epoch
                self._logger.store(**{'Time/Update': time.time() - epoch_time})
                eval_time = time.time()
                self._evaluate()

                self._logger.store(
                    {
                        'Time/Evaluate': time.time() - eval_time,
                        'Time/Epoch': time.time() - epoch_time,
                        'Time/Total': time.time() - start_time,
                        'Train/Epoch': self.epoch,
                        'TotalSteps': step + 1,
                    },
                )

                epoch_time = time.time()
                self._logger.dump_tabular()

                # save model to disk
                if self.epoch % self._cfgs.logger_cfgs.save_model_freq == 0:
                    self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    @abstractmethod
    def _train(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> None:
        """Train the model."""

    def _evaluate(self) -> None:
        """Evaluate the model."""
        self._env.evaluate(
            evaluate_epoisodes=self._cfgs.train_cfgs.evaluate_epoisodes,
            logger=self._logger,
            agent=self._actor,
        )
