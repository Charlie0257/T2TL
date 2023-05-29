from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy as np
from collections import deque

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 history_length):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc  # 4096
        self.discount = discount  # 0.998
        self.lr = lr  # 0.0003
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef  # 0.95
        self.value_loss_coef = value_loss_coef  # 0.5
        self.max_grad_norm = max_grad_norm  # 0.5
        self.recurrence = recurrence  # 1
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward  # None
        self.action_space_shape = envs[0].action_space.shape  # 2
        self.use_cont = self.acmodel.context


        # Control parameters
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)  # 16
        self.num_frames = self.num_frames_per_proc * self.num_procs # 4096*16=65536

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)  # shape: (4096, 16)
        act_shape = shape + self.action_space_shape  # act_shape: (4096, 16, 2)

        # in this, each env has its own original ltl
        self.obs = self.env.reset()  # all 16 envs will be reset (in this, each env has its own original ltl)
        self.obss = [None]*(shape[0])  # [None,...4096..., None]
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)  # [1., ...16..., 1.]
        self.masks = torch.zeros(*shape, device=self.device)  # torch.Size([4096, 16])
        self.actions = torch.zeros(*act_shape, device=self.device)  #, dtype=torch.int) torch.Size([4096, 16, 2])
        self.values = torch.zeros(*shape, device=self.device)  # torch.Size([4096, 16])
        self.rewards = torch.zeros(*shape, device=self.device)  # torch.Size([4096, 16])
        self.advantages = torch.zeros(*shape, device=self.device)  # torch.Size([4096, 16])
        self.log_probs = torch.zeros(*act_shape, device=self.device)  # torch.Size([4096, 16, 2])

        if self.use_cont:
            # Initialize Context Variable Setup  # todo: check
            ### history ####
            with torch.no_grad():
                preprocess_obs = self.preprocess_obss(self.obs, device=self.device)
                preprocess_obs = self.acmodel.env_model(preprocess_obs)
            self.history_length = history_length
            self.rewards_hist = deque(maxlen=history_length)
            self.actions_hist = deque(maxlen=history_length)
            self.obsvs_hist = deque(maxlen=history_length)

            self.next_hrews = deque(maxlen=history_length)
            self.next_hacts = deque(maxlen=history_length)
            self.next_hobvs = deque(maxlen=history_length)

            zero_action = torch.zeros(*(self.num_procs, self.action_space_shape[0]), device=self.device)
            zero_obs = torch.zeros(*(self.num_procs, (self.acmodel.embedding_size-self.acmodel.text_embedding_size)), device=self.device)
            for _ in range(history_length):
                self.rewards_hist.append(torch.zeros(*(shape[1], 1), device=self.device))
                self.actions_hist.append(zero_action.clone())
                self.obsvs_hist.append(zero_obs.clone())

                # same thing for next_h*
                self.next_hrews.append(torch.zeros(*(shape[1], 1), device=self.device))
                self.next_hacts.append(zero_action.clone())
                self.next_hobvs.append(zero_obs.clone())

            self.rewards_hist.append(torch.zeros(*(shape[1], 1), device=self.device))
            self.obsvs_hist.append(preprocess_obs.clone())

            rand_action = torch.FloatTensor(envs[0].action_space.sample()).unsqueeze(0)
            for m in range(len(envs)-1):
                rand_action = torch.concat([rand_action,
                                            torch.FloatTensor(envs[m+1].action_space.sample()).unsqueeze(0)], dim=0)
            self.actions_hist.append(rand_action.to(self.device).clone())

            self.rewards_hist_pro = torch.zeros(*(shape[0], history_length*shape[1]), device=self.device)
            self.actions_hist_pro = torch.zeros(*(shape[0], history_length*self.action_space_shape[0]*shape[1]), device=self.device)
            self.obsvs_hist_pro = torch.zeros(*(shape[0],
                                                history_length*shape[1]*(self.acmodel.embedding_size-self.acmodel.text_embedding_size)),
                                              device=self.device)

            self.next_hrews_pro = torch.zeros(*(shape[0], history_length*shape[1]), device=self.device)
            self.next_hacts_pro = torch.zeros(*(shape[0], history_length*self.action_space_shape[0]*shape[1]), device=self.device)
            self.next_hobvs_pro = torch.zeros(*(shape[0],
                                                history_length*shape[1]*(self.acmodel.embedding_size-self.acmodel.text_embedding_size)),
                                              device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)  # shape = (16,)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)  # shape = (16,)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)  # shape = (16,)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """
        Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):  # range(4096)
            # Do one agent-environment interaction

            if self.use_cont:
                # previous context variable
                np_pre_actions, np_pre_rewards, np_pre_obsers = self.actions_hist[0], self.rewards_hist[0], self.obsvs_hist[0]
                for k in range(self.history_length-1):
                    np_pre_actions = torch.concat([np_pre_actions, self.actions_hist[k + 1]], dim=1)
                    np_pre_rewards = torch.concat([np_pre_rewards, self.rewards_hist[k + 1]], dim=1)
                    np_pre_obsers = torch.concat([np_pre_obsers, self.obsvs_hist[k + 1]], dim=1)

                self.actions_hist_pro[i] = np_pre_actions.flatten().unsqueeze(0)
                self.rewards_hist_pro[i] = np_pre_rewards.flatten().unsqueeze(0)
                self.obsvs_hist_pro[i] = np_pre_obsers.flatten().unsqueeze(0)

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                elif self.acmodel.context:
                    dist, value, embedding = self.acmodel(preprocessed_obs,
                                                          [np_pre_actions, np_pre_rewards, np_pre_obsers],
                                                          )
                else:
                    dist, value = self.acmodel(preprocessed_obs)  # dist = Normal(loc: torch.Size([16, 2]), scale: torch.Size([16, 2])); shape(value)=16
            action = dist.sample()  # shape = torch.Size([16, 2])

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            if self.use_cont:
                ###############
                self.next_hrews.append(torch.FloatTensor(reward).view(self.num_procs, 1).to(self.device))
                self.next_hacts.append(action.clone())
                self.next_hobvs.append(embedding.clone())  # todo: check

                # np_next_hacts and np_next_hrews are required for TD3 alg
                np_next_hacts, np_next_hrews, np_next_hobvs = self.next_hacts[0], self.next_hrews[0], self.next_hobvs[0]
                for k in range(self.history_length - 1):
                    np_next_hacts= torch.concat([np_next_hacts, self.next_hacts[k + 1]], dim=1)
                    np_next_hrews = torch.concat([np_next_hrews, self.next_hrews[k + 1]], dim=1)
                    np_next_hobvs= torch.concat([np_next_hobvs, self.next_hobvs[k + 1]], dim=1)
                # np_next_hacts = np.asarray(self.next_hacts, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
                # np_next_hrews = np.asarray(self.next_hrews, dtype=np.float32)  # (hist, )
                # np_next_hobvs = np.asarray(self.next_hobvs, dtype=np.float32).flatten()  # (hist, )
                self.next_hacts_pro[i] = np_next_hacts.flatten().unsqueeze(0)
                self.next_hrews_pro[i] = np_next_hrews.flatten().unsqueeze(0)
                self.next_hobvs_pro[i] = np_next_hobvs.flatten().unsqueeze(0)

                # new becomes old
                self.rewards_hist.append(torch.FloatTensor(reward).view(self.num_procs, 1).to(self.device))
                self.actions_hist.append(action.clone())
                self.obsvs_hist.append(embedding.clone())  # todo: check

            # Update experiences values
            self.obss[i] = self.obs  # each i = {list: 16}
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            elif self.acmodel.context:
                _, next_value, _ = self.acmodel(preprocessed_obs,
                                                [np_next_hacts, np_next_hrews, np_next_hobvs],
                                                )
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape((-1, ) + self.action_space_shape)

        if self.use_cont:
            exps.actions_hist = self.actions_hist_pro.reshape(-1, self.history_length*self.action_space_shape[0])
            exps.rewards_hist = self.rewards_hist_pro.reshape(-1, self.history_length)
            exps.obsvs_hist = self.obsvs_hist_pro.reshape(-1, self.history_length*(self.acmodel.embedding_size-self.acmodel.text_embedding_size))

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
