import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributions.kl as kl
from cerml.utils import generate_gaussian
import rlkit.torch.pytorch_util as ptu

from rlkit.core import logger

import matplotlib.pyplot as plt

# -------------------

# This code is based on rlkit sac_v2 implementation.

from collections import OrderedDict

import gtimer as gt

from rlkit.core.eval_util import create_stats_ordered_dict



class CombinationTrainer:
    def __init__(self,
                 # from reconstruction trainer
                 encoder,
                 decoder,
                 prior_pz_layer,
                 replay_buffer,
                 batch_size,
                 num_classes,
                 latent_dim,
                 lr_decoder,
                 lr_encoder,
                 alpha_kl_z,
                 beta_kl_y,
                 use_state_diff,
                 state_reconstruction_clip,
                 factor_qf_loss,
                 train_val_percent,
                 eval_interval,
                 early_stopping_threshold,
                 temp_folder,

                 # from policy trainer
                 policy,
                 qf1,
                 qf2,
                 target_qf1,
                 target_qf2,
                 env_action_space,
                 discount=0.99,
                 reward_scale=1.0,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 optimizer_class=optim.Adam,
                 soft_target_tau=5e-3,
                 target_update_period=1,
                 plotter=None,
                 render_eval_paths=False,
                 use_automatic_entropy_tuning=True,
                 target_entropy=None,
                 target_entropy_factor=1.0
    ):
        super(CombinationTrainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior_pz_layer = prior_pz_layer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.lr_decoder = lr_decoder
        self.lr_encoder = lr_encoder
        self.alpha_kl_z = alpha_kl_z
        self.beta_kl_y = beta_kl_y
        self.use_state_diff = use_state_diff
        self.state_reconstruction_clip = state_reconstruction_clip
        self.factor_qf_loss = factor_qf_loss
        self.train_val_percent = train_val_percent
        self.eval_interval = eval_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.temp_folder = temp_folder

        self.factor_state_loss = 1
        self.factor_reward_loss = self.state_reconstruction_clip

        self.loss_weight_state = self.factor_state_loss / (self.factor_state_loss + self.factor_reward_loss)
        self.loss_weight_reward = self.factor_reward_loss / (self.factor_state_loss + self.factor_reward_loss)
        self.loss_weight_qf_loss = factor_qf_loss

        self.lowest_loss = np.inf
        self.lowest_loss_epoch = 0

        self.encoder_path = os.path.join(os.getcwd(), self.temp_folder, 'encoder.pth')
        self.decoder_path = os.path.join(os.getcwd(), self.temp_folder, 'decoder.pth')
        self.qf1_path = os.path.join(os.getcwd(), self.temp_folder, 'qf1.pth')
        self.qf2_path = os.path.join(os.getcwd(), self.temp_folder, 'qf2.pth')

        self.optimizer_class = optimizer_class

        self.loss_state_decoder = nn.MSELoss()
        self.loss_reward_decoder = nn.MSELoss()

        self.optimizer_encoder = self.optimizer_class(
            self.encoder.parameters(),
            lr=self.lr_encoder,
        )

        self.optimizer_decoder = self.optimizer_class(
            self.decoder.parameters(),
            lr=self.lr_decoder,
        )

        # --------------

        self.env_action_space = env_action_space
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -self.env_action_space  # heuristic value from Tuomas
            self.target_entropy = self.target_entropy * target_entropy_factor
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        # added here because needed in Q function optimization
        self.alpha = self.log_alpha.exp()

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, epochs, w_method="val_value_based"):
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        train_overall_losses = []
        train_state_losses = []
        train_reward_losses = []
        train_qf_losses = []

        policy_losses = []
        alphas = []
        log_pis = []

        train_val_state_losses = []
        train_val_reward_losses = []
        train_val_qf_losses = []
        train_val_qf1_losses = []
        train_val_qf2_losses = []

        val_state_losses = []
        val_reward_losses = []
        val_qf_losses = []
        val_qf1_losses = []
        val_qf2_losses = []

        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        for epoch in range(epochs):
            overall_loss, state_loss, reward_loss, policy_loss, alpha, log_pi, qf_loss, qf1_loss, qf2_loss = self.training_step(train_indices)
            train_overall_losses.append(overall_loss)
            train_state_losses.append(state_loss)
            train_reward_losses.append(reward_loss)
            train_qf_losses.append(qf_loss)
            policy_losses.append(policy_loss / 1.0)
            alphas.append(alpha / 1.0)
            log_pis.append((-1) * log_pi.mean() / 1.0)
            if epoch % 100 == 0 and int(os.environ['DEBUG']) == 1:
                print("Epoch: " + str(epoch) + ", policy loss: " + str(policy_losses[-1]))

            #Evaluate with validation set for early stopping
            if epoch % self.eval_interval == 0:
                val_state_loss, val_reward_loss, val_qf_loss, val_qf1_loss, val_qf2_loss = self.validate(val_indices)
                val_state_losses.append(val_state_loss)
                val_reward_losses.append(val_reward_loss)
                val_qf_losses.append(val_qf_loss)
                val_qf1_losses.append(val_qf1_loss)
                val_qf2_losses.append(val_qf2_loss)
                train_val_state_loss, train_val_reward_loss, train_val_qf_loss, train_val_qf1_loss, train_val_qf2_loss = self.validate(train_indices)
                train_val_state_losses.append(train_val_state_loss)
                train_val_reward_losses.append(train_val_reward_loss)
                train_val_qf_losses.append(train_val_qf_loss)
                train_val_qf1_losses.append(train_val_qf1_loss)
                train_val_qf2_losses.append(train_val_qf2_loss)

                # change loss weighting
                if w_method == "val_value_based":
                    weight_factors = np.ones(2)
                    weights = np.array([train_val_state_loss * self.factor_state_loss, train_val_reward_loss * self.factor_reward_loss])
                    for i in range(weights.shape[0]):
                        weight_factors[i] = weights[i] / np.sum(weights)
                    self.loss_weight_state = weight_factors[0]
                    self.loss_weight_reward = weight_factors[1]
                    if int(os.environ['DEBUG']) == 1:
                        print("weight factors: " + str(weight_factors))
                if int(os.environ['DEBUG']) == 1:
                    print("\nEpoch: " + str(epoch))
                    #print("State loss: " + str(train_state_losses[-1]))
                    #print("Reward loss: " + str(train_reward_losses[-1]))
                    print("Overall loss: " + str(train_overall_losses[-1]))
                    print("Train Validation loss (state, reward): " + str(train_val_state_losses[-1]) + ' , ' + str(train_val_reward_losses[-1]))
                    print("Validation loss (state, reward): " + str(val_state_losses[-1]) + ' , ' + str(val_reward_losses[-1]))
                    print("Qf loss (train, val):" + str(train_val_qf_losses[-1]) + ' , ' + str(val_qf_losses[-1]))
                if self.early_stopping(epoch, val_state_loss + val_reward_loss + val_qf_loss):
                    print("Early stopping at epoch " + str(epoch))
                    break

        a = self.validate(train_indices)
        b = self.validate(val_indices)

        # load the least loss encoder
        #self.encoder.load_state_dict(torch.load(self.encoder_path))
        #self.decoder.load_state_dict(torch.load(self.decoder_path))
        #self.qf1.load_state_dict(torch.load(self.qf1_path))
        #self.qf2.load_state_dict(torch.load(self.qf2_path))
        if int(os.environ['PLOT']) == 1:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(list(range(len(train_overall_losses))), np.array(train_overall_losses), label="Train overall loss")
            plt.xlim(left=0)
            #plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(list(range(len(train_state_losses))), np.array(train_state_losses) + np.array(train_reward_losses), label="Train loss without KL terms")
            plt.xlim(left=0)
            #plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(np.array(list(range(len(train_val_state_losses)))) * self.eval_interval, np.array(train_val_state_losses), label="train_val_state_losses")
            plt.plot(np.array(list(range(len(train_val_reward_losses)))) * self.eval_interval, np.array(train_val_reward_losses), label="train_val_reward_losses")
            plt.plot(np.array(list(range(len(val_state_losses)))) * self.eval_interval, np.array(val_state_losses), label="val_state_losses")
            plt.plot(np.array(list(range(len(val_reward_losses)))) * self.eval_interval, np.array(val_reward_losses), label="val_reward_losses")
            plt.xlim(left=0)
            # plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()

            plt.show()

            plt.figure()
            plt.plot(np.array(list(range(len(train_qf_losses)))), np.array(train_qf_losses), label="Train Qf loss")
            plt.plot(np.array(list(range(len(train_val_qf_losses)))) * self.eval_interval, np.array(train_val_qf_losses), label="Train Val Qf loss")
            plt.plot(np.array(list(range(len(val_qf_losses)))) * self.eval_interval, np.array(val_qf_losses), label="Val Qf loss")
            plt.plot(np.array(list(range(len(val_qf1_losses)))) * self.eval_interval, np.array(val_qf1_losses), label="Val Qf1 loss")
            plt.plot(np.array(list(range(len(val_qf2_losses)))) * self.eval_interval, np.array(val_qf2_losses), label="Val Qf2 loss")
            plt.xlim(left=0)
            # plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()
            plt.show()

        #just for debug
        #self.training_step(train_indices)
        a = self.validate(train_indices)
        b = self.validate(val_indices)

        logger.record_tabular("Reconstruction_train_val_state_loss", a[0])
        logger.record_tabular("Reconstruction_train_val_reward_loss", a[1])
        logger.record_tabular("Reconstruction_train_val_qf_loss", a[2])
        logger.record_tabular("Reconstruction_val_state_loss", b[0])
        logger.record_tabular("Reconstruction_val_reward_loss", b[1])
        logger.record_tabular("Reconstruction_val_qf_loss", b[2])
        logger.record_tabular("Reconstruction_epochs", epoch + 1)


        # -----------------



        if int(os.environ['PLOT']) == 1:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(list(range(len(policy_losses))), np.array(policy_losses), label="Policy loss")
            plt.xlim(left=0)
            plt.legend()
            # plt.ylim(bottom=0)
            plt.subplot(3, 1, 2)
            plt.plot(list(range(len(alphas))), np.array(alphas), label="alphas")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(list(range(len(log_pis))), np.array(log_pis), label="Entropy")
            plt.legend()
            plt.show(block=False)

        self.eval_statistics['policy_train_steps_total'] = self._n_train_steps_total
        self.end_epoch(epoch)

        return policy_losses[-1], self.get_diagnostics()


    def training_step(self, indices):
        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder.
        The overall objective due to the generative model is:
        parameter* = arg max ELBO
        ELBO = sum_k q(y=k | x) * [ log p(x|z_k) - KL ( q(z, x,y=k) || p(z|y=k) ) ] - KL ( q(y|x) || p(y) )
        '''
        # get data from replay buffer
        # TODO: for validation data use all data --> batch size == validation size
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=False)
        #data = self.replay_buffer.sample_random_few_step_batch(np.array([3,4,5,6]), self.batch_size, normalize=False)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        # prepare for usage in decoder
        actions = ptu.from_numpy(data['actions'])[:, -1, :]
        states = ptu.from_numpy(data['observations'])[:, -1, :]
        next_states = ptu.from_numpy(data['next_observations'])[:, -1, :]
        rewards = ptu.from_numpy(data['rewards'])[:, -1, :]
        terminals = ptu.from_numpy(data['terminals'])[:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (next_states - states)[:, :self.state_reconstruction_clip]
        else:
            decoder_state_target = next_states[:, :self.state_reconstruction_clip]

        # Forward pass through encoder
        y_distribution, z_distributions = self.encoder.encode(encoder_input)

        kl_qz_pz = ptu.zeros(self.batch_size, self.num_classes)
        state_losses = ptu.zeros(self.batch_size, self.num_classes)
        reward_losses = ptu.zeros(self.batch_size, self.num_classes)
        qf_losses = ptu.zeros(self.batch_size, self.num_classes)
        qf1_losses = ptu.zeros(self.batch_size, self.num_classes)
        qf2_losses = ptu.zeros(self.batch_size, self.num_classes)
        nll_px = ptu.zeros(self.batch_size, self.num_classes)

        # every y component (see ELBO formula)
        # TODO: write in matrix form directly, probably not possible because no option to stack distributions
        for y in range(self.num_classes):
            z = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y)

            # put in decoder to get likelihood
            state_estimate, reward_estimate = self.decoder(states, actions, next_states, z)
            state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
            reward_loss = torch.sum((reward_estimate - rewards) ** 2, dim=1)
            state_losses[:, y] = state_loss
            reward_losses[:, y] = reward_loss

            """
            QF Loss
            """
            obs = torch.cat((states, z), dim=1)
            next_obs = torch.cat((next_states, z), dim=1)

            q1_pred = self.qf1(obs, actions)
            q2_pred = self.qf2(obs, actions)
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - self.alpha * new_log_pi

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            #qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            #qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
            qf1_loss = torch.sum((q1_pred - q_target.detach()) ** 2, dim=1)
            qf2_loss = torch.sum((q2_pred - q_target.detach()) ** 2, dim=1)
            qf_loss = 0.5 * (qf1_loss + qf2_loss)
            qf_losses[:, y] = qf_loss
            qf1_losses[:, y] = qf1_loss
            qf2_losses[:, y] = qf2_loss

            nll_px[:, y] = self.loss_weight_state * state_loss + self.loss_weight_reward * reward_loss + self.loss_weight_qf_loss * qf_loss

            # KL ( q(z, x,y=k) || p(z|y=k) )
            prior = self.prior_pz(y)
            kl_qz_pz[:, y] = torch.sum(kl.kl_divergence(z_distributions[y], prior), dim=1)

        kl_qy_py = kl.kl_divergence(y_distribution, self.prior_py()).view(-1, 1)

        # Overall ELBO
        elbo = torch.sum(torch.sum(torch.mul(y_distribution.probs,  (-1) * nll_px - self.alpha_kl_z * kl_qz_pz)) - self.beta_kl_y * kl_qy_py)
        # but elbo should be maximized, and backward function assumes minimization
        loss = (-1) * elbo

        # Optimization strategy:
        # Decoder: the two head loss functions backpropagate their gradients into corresponding parts
        # of the network, then ONE common optimizer compute all weight updates
        # Encoder: the KLs and the likelihood from the decoder backpropagate their gradients into
        # corresponding parts of the network, then ONE common optimizer computes all weight updates
        # This is not done explicitly but all within the elbo function

        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.optimizer_decoder.step()
        self.optimizer_encoder.step()

        include_policy = False
        policy_loss = ptu.zeros(self.batch_size)
        log_pi = ptu.zeros(self.batch_size)
        if include_policy:
            # ------ after encoder, decoder is updated, we apply regular SAC with best encoder guess
            # compute most likely z (like it would be computed while test time or done by the relabeler)
            z = self.encoder(encoder_input)

            # detach, as we only want gradients from Q into the encoder
            obs = torch.cat((states, z.detach()), dim=1)
            next_obs = torch.cat((next_states, z.detach()), dim=1)

            """
            Policy and Alpha Loss
            """
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                # alpha = self.log_alpha.exp()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                self.alpha = 1
            # alpha = alpha.to(ptu.device)
            # self.alpha = self.alpha.to(ptu.device)

            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )
            policy_loss = (self.alpha * log_pi - q_new_actions).mean()

            """
            QF Loss
            """
            q1_pred = self.qf1(obs, actions)
            q2_pred = self.qf2(obs, actions)
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - self.alpha * new_log_pi

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

            """
            Update networks
            """
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1, self.target_qf1, self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2, self.target_qf2, self.soft_target_tau
                )

            """
            Save some statistics for eval
            """
            if self._need_to_update_eval_statistics:
                self._need_to_update_eval_statistics = False
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                policy_loss = (log_pi - q_new_actions).mean()

                self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                    policy_loss
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q1 Predictions',
                    ptu.get_numpy(q1_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Targets',
                    ptu.get_numpy(q_target),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
                if self.use_automatic_entropy_tuning:
                    self.eval_statistics['Alpha'] = self.alpha.item()
                    self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

        #if policy_loss is None:
        #    policy_loss = ptu.zeros(self.batch_size)
        #if log_pi is None:
        #    log_pi = ptu.zeros(self.batch_size)

        return ptu.get_numpy(loss)/self.batch_size,\
               ptu.get_numpy(torch.sum(state_losses, axis=0))/self.batch_size,\
               ptu.get_numpy(torch.sum(reward_losses, axis=0))/self.batch_size,\
               ptu.get_numpy(policy_loss),\
               ptu.get_numpy(self.alpha),\
               ptu.get_numpy(log_pi),\
               ptu.get_numpy(torch.sum(qf_losses, axis=0))/self.batch_size,\
               ptu.get_numpy(torch.sum(qf1_losses, axis=0))/self.batch_size,\
               ptu.get_numpy(torch.sum(qf2_losses, axis=0))/self.batch_size

    def prior_pz(self, y):
        '''
        As proposed in the CURL paper: use linear layer, that conditioned on y gives Gaussian parameters
        '''
        one_hot = ptu.zeros(self.batch_size, self.num_classes)
        one_hot[:, y] = 1
        mu_sigma = self.prior_pz_layer(one_hot).detach() # we do not want to backprop into prior
        #for debug
        return torch.distributions.normal.Normal(ptu.ones(self.batch_size, 1) * y, ptu.ones(self.batch_size, 1) * 0.5)

        if self.encoding_neglect_z:
            mu_sigma = ptu.ones(self.batch_size, self.latent_dim * 2)
            mu_sigma[:, 0] = mu_sigma[:, 0] * y
            mu_sigma[:, 1] = mu_sigma[:, 1] * 0.01

            return generate_gaussian(mu_sigma, self.latent_dim, sigma_ops=None)

        # mu_sigma = ptu.ones(self.batch_size, self.latent_dim * 2)
        # mu_sigma[:, 0] = mu_sigma[:, 0] * y - 0.5
        # mu_sigma[:, 1] = mu_sigma[:, 1] * 0.1
        else:
            return generate_gaussian(mu_sigma, self.latent_dim)

    def prior_py(self):
        '''
        Categorical uniform distribution
        '''
        return torch.distributions.categorical.Categorical(probs=ptu.ones(self.batch_size, self.num_classes) * (1.0/self.num_classes))

    def validate(self, indices):
        # get data from replay buffer
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=False)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        # prepare for usage in decoder
        actions = ptu.from_numpy(data['actions'])[:, -1, :]
        states = ptu.from_numpy(data['observations'])[:, -1, :]
        next_states = ptu.from_numpy(data['next_observations'])[:, -1, :]
        rewards = ptu.from_numpy(data['rewards'])[:, -1, :]
        terminals = ptu.from_numpy(data['terminals'])[:, -1, :]
        debug_true_tasks = ptu.from_numpy(data['true_tasks'])

        if self.use_state_diff:
            decoder_state_target = (next_states - states)[:, :self.state_reconstruction_clip]
        else:
            decoder_state_target = next_states[:, :self.state_reconstruction_clip]

        z = self.encoder(encoder_input)
        #z = torch.rand_like(z)

        # decoder part
        state_estimate, reward_estimate = self.decoder(states, actions, next_states, z)
        state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
        reward_loss = torch.sum((reward_estimate - rewards) ** 2, dim=1)

        # Q function part
        """
        QF Loss
        """
        obs = torch.cat((states, z), dim=1)
        next_obs = torch.cat((next_states, z), dim=1)

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - self.alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        #qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        #qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        qf1_loss = torch.sum((q1_pred - q_target.detach()) ** 2, dim=1)
        qf2_loss = torch.sum((q2_pred - q_target.detach()) ** 2, dim=1)
        qf_loss = 0.5 * (qf1_loss + qf2_loss)


        return ptu.get_numpy(torch.sum(state_loss)) / self.batch_size,\
               ptu.get_numpy(torch.sum(reward_loss))/ self.batch_size, \
               ptu.get_numpy(torch.sum(qf_loss)) / self.batch_size,\
               ptu.get_numpy(torch.sum(qf1_loss)) / self.batch_size,\
               ptu.get_numpy(torch.sum(qf2_loss)) / self.batch_size


    def early_stopping(self, epoch, loss):
        if loss < self.lowest_loss:
            if int(os.environ['DEBUG']) == 1:
                print("Found new minimum at Epoch " + str(epoch))
            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch
            torch.save(self.encoder.state_dict(), self.encoder_path)
            torch.save(self.decoder.state_dict(), self.decoder_path)
            torch.save(self.qf1.state_dict(), self.qf1_path)
            torch.save(self.qf2.state_dict(), self.qf2_path)
        if epoch - self.lowest_loss_epoch > self.early_stopping_threshold:
            return True
        else:
            return False

    # ----------------------

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
