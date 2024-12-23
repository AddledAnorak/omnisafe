"""Implementation of Shoan's Offline Algorithm."""

from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.base import Actor, Critic
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor

'''
Configs required by code:

algo_cfgs:
    sampled_action_num: int
    minimum_weighting: float
    batch_size: int
    gamma: float
    cost_limit: float
    bc_coeff: float

model_cfgs:
    actor:
        hidden_sizes: int
        activation: str
        lr: float
    
    critic:
        hidden_sizes: int
        activation: str
        lr: float
    
    weight_initialization_mode



'''


@registry.register
class OfflineShoan(BaseOffline):
    def _init_log(self):
        super()._init_log()
        what_to_save: Dict[str, Any] = {
            'actor': self._actor,
        }
        self._logger.setup_torch_saver(what_to_save)

        self._logger.register_key('Loss/Loss_reward_critic')
        self._logger.register_key('Qr/data_Qr')
        self._logger.register_key('Qr/target_Qr')

        self._logger.register_key('Loss/Loss_cost_critic')
        self._logger.register_key('Qc/data_Qc')
        self._logger.register_key('Qc/target_Qc')

        self._logger.register_key('Loss/BC_term_loss')
        self._logger.register_key('Loss/Loss_actor')
        self._logger.register_key('Actor/Ratio_safe_to_total_actions')
        self._logger.register_key('Actor/Ratio_unsafe_to_total_actions')
        
    


    def _init_model(self):
        super()._init_model()

        self._actor: Actor = GaussianLearningActor(
            self._env.observation_space,
            self._env.action_space,
            self._cfgs.model_cfgs.actor.hidden_sizes,
            activation=self._cfgs.model_cfgs.actor.activation,
            weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
        )
        assert isinstance(
            self._cfgs.model_cfgs.actor.lr,
            float,
        ), 'The learning rate of actor must be a float number.'
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._cfgs.model_cfgs.actor.lr,
        )


        self._reward_critic: Critic = (
            CriticBuilder(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.critic.hidden_sizes,
                activation=self._cfgs.model_cfgs.critic.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
                num_critics=2,
            )
            .build_critic('q')
            .to(self._device)
        )
        self._target_reward_critic: Critic = deepcopy(self._reward_critic)
        assert isinstance(
            self._cfgs.model_cfgs.critic.lr,
            float,
        ), 'The learning rate of critic must be a float number.'
        self._reward_critic_optimizer = optim.Adam(
            self._reward_critic.parameters(),
            lr=self._cfgs.model_cfgs.critic.lr,
        )

        self._cost_critic: Critic = (
            CriticBuilder(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.critic.hidden_sizes,
                activation=self._cfgs.model_cfgs.critic.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
                num_critics=2,
            )
            .build_critic('q')
            .to(self._device)
        )
        self._target_cost_critic: Critic = deepcopy(self._reward_critic)
        assert isinstance(
            self._cfgs.model_cfgs.critic.lr,
            float,
        ), 'The learning rate of critic must be a float number.'
        self._cost_critic_optimizer = optim.Adam(
            self._cost_critic.parameters(),
            lr=self._cfgs.model_cfgs.critic.lr,
        )


    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        obs, action, reward, cost, next_obs, done = batch

        self._update_reward_critic(obs, action, reward, next_obs, done)
        self._update_cost_critic(obs, action, cost, next_obs, done)
        self._update_actor(obs, action)

        self._polyak_update()

    
    def _update_reward_critic(
        self, 
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        with torch.no_grad():
            next_obs_repeat = torch.repeat_interleave(
                next_obs,
                self._cfgs.algo_cfgs.sampled_action_num,
                dim=0,
            )
            next_actions = self._actor.predict(next_obs_repeat)

            future_q1, future_q2 = self._target_reward_critic.forward(next_obs_repeat, next_actions)
            q_target = self._cfgs.algo_cfgs.minimum_weighting * torch.min(
                future_q1,
                future_q2,
            ) + (1 - self._cfgs.algo_cfgs.minimum_weighting) * torch.max(future_q1, future_q2)
            q_target = (
                q_target.reshape(self._cfgs.algo_cfgs.batch_size, -1).max(dim=1)[0].reshape(-1)
            )
            q_target = reward + (1 - done) * self._cfgs.algo_cfgs.gamma * q_target
        
        q1, q2 = self._reward_critic.forward(obs, action)

        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(
            q2,
            q_target,
        )

        self._reward_critic_optimizer.zero_grad()
        critic_loss.backward()
        self._reward_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_reward_critic': critic_loss.item(),
                'Qr/data_Qr': q1[0].mean().item(),
                'Qr/target_Qr': q_target[0].mean().item(),
            },
        )


    def _update_cost_critic(
        self, 
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        with torch.no_grad():
            next_obs_repeat = torch.repeat_interleave(
                next_obs,
                self._cfgs.algo_cfgs.sampled_action_num,
                dim=0,
            )
            next_actions = self._actor.predict(next_obs_repeat)
            future_q1, future_q2 = self._target_cost_critic.forward(next_obs_repeat, next_actions)
            q_target = self._cfgs.algo_cfgs.minimum_weighting * torch.max(
                future_q1,
                future_q2,
            ) + (1 - self._cfgs.algo_cfgs.minimum_weighting) * torch.min(future_q1, future_q2)
            q_target = (
                q_target.reshape(self._cfgs.algo_cfgs.batch_size, -1).max(dim=1)[0].reshape(-1)
            )
            q_target = reward + (1 - done) * self._cfgs.algo_cfgs.gamma * q_target
        
        q1, q2 = self._cost_critic.forward(obs, action)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(
            q2,
            q_target,
        )

        self._cost_critic_optimizer.zero_grad()
        critic_loss.backward()
        self._cost_critic_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_cost_critic': critic_loss.item(),
                'Qc/data_Qc': q1[0].mean().item(),
                'Qc/target_Qc': q_target[0].mean().item(),
            },
        )


    def _update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> None:
        obs_repeat = torch.repeat_interleave(
            obs,
            self._cfgs.algo_cfgs.sampled_action_num,
            dim=0,
        )

        act_repeat = torch.repeat_interleave(
            action,
            self._cfgs.algo_cfgs.sampled_action_num,
            dim=0,
        )

        actions = self._actor.predict(obs_repeat)
        qr1, qr2 = self._reward_critic.forward(obs_repeat, actions) # (N, batch_size, 1)
        qc1, qc2 = self._cost_critic.forward(obs_repeat, actions)

        qr = (self._cfgs.algo_cfgs.minimum_weighting * torch.min(qr1, qr2)
            + (1 - self._cfgs.algo_cfgs.minimum_weighting) * torch.max(qr1, qr2))
        
        qc = (self._cfgs.algo_cfgs.minimum_weighting * torch.max(qc1, qc2)
            + (1 - self._cfgs.algo_cfgs.minimum_weighting) * torch.min(qc1, qc2))
        
        cost_limit = self._cfgs.algo_cfgs.cost_limit

        def safe_loss(qr_safe: torch.Tensor) -> torch.Tensor:
            # let shape of qr_safe be (n)
            return -qr_safe.sum()
    
        def unsafe_loss(qc_unsafe: torch.Tensor) -> torch.Tensor:
            return qc_unsafe.sum()
        

        bc_term_loss = self._cfgs.algo_cfgs.bc_coeff * ((act_repeat - actions) ** 2).mean() / self._cfgs.algo_cfgs.sampled_action_num
        # loss = bc_term_loss + sum([ 
        #     (safe_loss(qr[:, i][qc[:, i] <= cost_limit]) + unsafe_loss(qc[:, i][qc[:, i] > cost_limit])) 
        #     for i in range(self._cfgs.algo_cfgs.batch_size)
        # ]) / (self._cfgs.algo_cfgs.sampled_action_num * self._cfgs.algo_cfgs.batch_size)

        loss = bc_term_loss + torch.where(qc <= cost_limit, -qr, qc).mean()

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        self._logger.store(
            **{
                'Loss/BC_term_loss': bc_term_loss.item(),
                'Loss/Loss_actor': loss.item(),
                'Actor/Ratio_safe_to_total_actions': (qc <= cost_limit).sum() / (self._cfgs.algo_cfgs.batch_size * self._cfgs.algo_cfgs.sampled_action_num),
                'Actor/Ratio_unsafe_to_total_actions': (qc > cost_limit).sum() / (self._cfgs.algo_cfgs.batch_size * self._cfgs.algo_cfgs.sampled_action_num)
            }
        )


    def _polyak_update(self):
        for target_param, param in zip(
            self._target_reward_critic.parameters(),
            self._reward_critic.parameters(),
        ):
            target_param.data.copy_(
                self._cfgs.algo_cfgs.polyak * param.data
                + (1 - self._cfgs.algo_cfgs.polyak) * target_param.data,
            )

        for target_param, param in zip(
            self._target_cost_critic.parameters(),
            self._cost_critic.parameters(),
        ):
            target_param.data.copy_(
                self._cfgs.algo_cfgs.polyak * param.data
                + (1 - self._cfgs.algo_cfgs.polyak) * target_param.data,
            )