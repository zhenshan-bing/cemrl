import numpy as np
import torch
from collections import OrderedDict
from rlkit.core import logger
import gtimer as gt
import pickle
import os
import ray
import gc
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu


class CEMRLAlgorithm:
    def __init__(self,
                 replay_buffer,
                 rollout_coordinator,
                 reconstruction_trainer,
                 combination_trainer,
                 policy_trainer,
                 relabeler,
                 agent,
                 networks,
                 train_tasks,
                 test_tasks,

                 num_epochs,
                 num_reconstruction_steps,
                 num_policy_steps,
                 num_train_tasks_per_episode,
                 num_transitions_initial,
                 num_transistions_per_episode,
                 num_eval_trajectories,
                 showcase_every,
                 snapshot_gap,
                 num_showcase_deterministic,
                 num_showcase_non_deterministic,
                 use_relabeler,
                 use_combination_trainer,
                 experiment_log_dir,
                 latent_dim
                 ):
        self.replay_buffer = replay_buffer
        self.rollout_coordinator = rollout_coordinator
        self.reconstruction_trainer = reconstruction_trainer
        self.combination_trainer = combination_trainer
        self.policy_trainer = policy_trainer
        self.relabeler = relabeler
        self.agent = agent
        self.networks = networks

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.num_epochs = num_epochs
        self.num_reconstruction_steps = num_reconstruction_steps
        self.num_policy_steps = num_policy_steps
        self.num_transitions_initial = num_transitions_initial
        self.num_train_tasks_per_episode = num_train_tasks_per_episode
        self.num_transitions_per_episode = num_transistions_per_episode
        self.num_eval_trajectories = num_eval_trajectories
        self.use_relabeler = use_relabeler
        self.use_combination_trainer = use_combination_trainer
        self.experiment_log_dir = experiment_log_dir
        self.latent_dim = latent_dim

        self.showcase_every = showcase_every
        self.snapshot_gap = snapshot_gap
        self.num_showcase_deterministic = num_showcase_deterministic
        self.num_showcase_non_deterministic = num_showcase_non_deterministic

        self._n_env_steps_total = 0

    def train(self):
        params = self.get_epoch_snapshot()
        logger.save_itr_params(-1, params)
        previous_epoch_end = 0

        print("Collecting initial samples ...")
        if self.num_transitions_initial > 0:
            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(self.train_tasks, max_samples=self.num_transitions_initial)

        for epoch in gt.timed_for(range(self.num_epochs), save_itrs=True):
            tabular_statistics = OrderedDict()

            # 1. collect data with rollout coordinator
            print("Collecting samples ...")
            data_collection_tasks = np.random.permutation(self.train_tasks)[:self.num_train_tasks_per_episode]
            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(data_collection_tasks, max_samples=self.num_transitions_per_episode)
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            gt.stamp('data_collection')

            # replay buffer stats
            self.replay_buffer.stats_dict = self.replay_buffer.get_stats()

            if self.use_combination_trainer:
                # 2. combination trainer
                print("Combination Trainer ...")
                temp, sac_stats = self.combination_trainer.train(self.num_reconstruction_steps)
                tabular_statistics.update(sac_stats)
                gt.stamp('reconstruction_trainer')

                # 3. relabel the data regarding z with relabeler
                if self.use_relabeler:
                    self.relabeler.relabel()
                gt.stamp('relabeler')

                # 4. train policy via SAC with data from the replay buffer
                print("Policy Trainer ...")
                temp, sac_stats = self.policy_trainer.train(self.num_policy_steps)
                tabular_statistics.update(sac_stats)

                # alpha optimized through policy trainer should be used in combination trainer as well
                self.combination_trainer.alpha = self.policy_trainer.log_alpha.exp()

            else:
                # 2. encoder - decoder training with reconstruction trainer
                print("Reconstruction Trainer ...")
                self.reconstruction_trainer.train(self.num_reconstruction_steps)
                gt.stamp('reconstruction_trainer')

                # 3. relabel the data regarding z with relabeler
                if self.use_relabeler:
                    self.relabeler.relabel()
                gt.stamp('relabeler')

                # 4. train policy via SAC with data from the replay buffer
                print("Policy Trainer ...")
                temp, sac_stats = self.policy_trainer.train(self.num_policy_steps)
                tabular_statistics.update(sac_stats)
            gt.stamp('policy_trainer')

            # 5. Evaluation
            print("Evaluation ...")
            eval_output = self.rollout_coordinator.evaluate('train', data_collection_tasks, self.num_eval_trajectories, deterministic=True, animated=False, log=True)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories, deterministic=False, animated=False, log=True)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories, deterministic=True, animated=False, log=True)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            gt.stamp('evaluation')

            # 6. Showcase if wanted
            if self.showcase_every != 0 and epoch % self.showcase_every == 0:
                self.rollout_coordinator.evaluate('test', self.test_tasks[:5], self.num_showcase_deterministic, deterministic=True, animated=True, log=False)
                self.rollout_coordinator.evaluate('test', self.test_tasks[:5], self.num_showcase_non_deterministic, deterministic=False, animated=True, log=False)
            gt.stamp('showcase')

            # 7. Logging
            # Network parameters
            params = self.get_epoch_snapshot()
            logger.save_itr_params(epoch, params)

            if epoch in logger._snapshot_points:
                # store encoding
                encoding_storage = self.replay_buffer.check_enc()
                pickle.dump(encoding_storage,
                            open(os.path.join(self.experiment_log_dir, "encoding_" + str(epoch) + ".p"), "wb"))

                # replay stats dict
                pickle.dump(self.replay_buffer.stats_dict,
                            open(os.path.join(self.experiment_log_dir, "replay_buffer_stats_dict_" + str(epoch) + ".p"),
                                 "wb"))
            gt.stamp('logging')

            # 8. Time
            times_itrs = gt.get_times().stamps.itrs
            tabular_statistics['time_data_collection'] = times_itrs['data_collection'][-1]
            tabular_statistics['time_reconstruction_trainer'] = times_itrs['reconstruction_trainer'][-1]
            tabular_statistics['time_relabeler'] = times_itrs['relabeler'][-1]
            tabular_statistics['time_policy_trainer'] = times_itrs['policy_trainer'][-1]
            tabular_statistics['time_evaluation'] = times_itrs['evaluation'][-1]
            tabular_statistics['time_showcase'] = times_itrs['showcase'][-1]
            tabular_statistics['time_logging'] = times_itrs['logging'][-1]
            total_time = gt.get_times().total
            epoch_time = total_time - previous_epoch_end
            previous_epoch_end = total_time
            tabular_statistics['time_epoch'] = epoch_time
            tabular_statistics['time_total'] = total_time

            # other
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            tabular_statistics['epoch'] = epoch

            for key, value in tabular_statistics.items():
                logger.record_tabular(key, value)

            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        ray.shutdown()

    def get_epoch_snapshot(self):
        snapshot = OrderedDict()
        for name, net in self.networks.items():
            snapshot[name] = net.state_dict()
        return snapshot

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            self.networks[net].to(device)
        self.agent.to(device)

    def plot_encodings(self, epoch):
        encoding_storage = pickle.load(open(os.path.join(self.experiment_log_dir, "encoding_" + str(epoch) + ".p"), "rb"))
        base_tasks = list(encoding_storage.keys())
        fig, axes_tuple = plt.subplots(ncols=len(base_tasks), sharey=True)
        if len(base_tasks) == 1: axes_tuple = [axes_tuple]
        for i, base in enumerate(base_tasks):
            for dim in range(self.latent_dim):
                axes_tuple[i].errorbar(list(encoding_storage[base].keys()), [a['mean'][dim] for a in list(encoding_storage[base].values())], yerr=[a['std'][dim] for a in list(encoding_storage[base].values())], fmt="o")
            axes_tuple[i].plot(list(encoding_storage[base].keys()), [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x')
        plt.show()
