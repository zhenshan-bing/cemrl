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


class ReconstructionTrainer(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 prior_pz_layer,
                 replay_buffer,
                 batch_size,
                 num_classes,
                 latent_dim,
                 timesteps,
                 lr_decoder,
                 lr_encoder,
                 alpha_kl_z,
                 beta_kl_y,
                 use_state_diff,
                 component_constraint_learning,
                 state_reconstruction_clip,
                 train_val_percent,
                 eval_interval,
                 early_stopping_threshold,
                 experiment_log_dir,
                 prior_mode,
                 prior_sigma,
                 isIndividualY,
                 data_usage_reconstruction,
                 optimizer_class=optim.Adam,
                 ):
        super(ReconstructionTrainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior_pz_layer = prior_pz_layer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.lr_decoder = lr_decoder
        self.lr_encoder = lr_encoder
        self.alpha_kl_z = alpha_kl_z
        self.beta_kl_y = beta_kl_y
        self.use_state_diff = use_state_diff
        self.component_constraint_learning = component_constraint_learning
        self.state_reconstruction_clip = state_reconstruction_clip
        self.train_val_percent = train_val_percent
        self.eval_interval = eval_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.experiment_log_dir = experiment_log_dir
        self.prior_mode = prior_mode
        self.prior_sigma = prior_sigma
        self.isIndividualY = isIndividualY
        self.data_usage_reconstruction = data_usage_reconstruction

        self.factor_state_loss = 1
        self.factor_reward_loss = self.state_reconstruction_clip

        self.loss_weight_state = self.factor_state_loss / (self.factor_state_loss + self.factor_reward_loss)
        self.loss_weight_reward = self.factor_reward_loss / (self.factor_state_loss + self.factor_reward_loss)

        self.lowest_loss = np.inf
        self.lowest_loss_epoch = 0

        self.temp_path = os.path.join(os.getcwd(), '.temp', self.experiment_log_dir.split('/')[-1])
        self.encoder_path = os.path.join(self.temp_path, 'encoder.pth')
        self.decoder_path = os.path.join(self.temp_path, 'decoder.pth')
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.optimizer_class = optimizer_class

        self.loss_state_decoder = nn.MSELoss()
        self.loss_reward_decoder = nn.MSELoss()

        self.optimizer_encoder = self.optimizer_class(
            list(self.encoder.parameters()) + list(self.prior_pz_layer.parameters()),
            lr=self.lr_encoder,
        )

        self.optimizer_decoder = self.optimizer_class(
            self.decoder.parameters(),
            lr=self.lr_decoder,
        )

    def train(self, epochs):
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        if self.data_usage_reconstruction == "tree_sampling":
            train_indices = np.random.permutation(train_indices)
            val_indices = np.random.permutation(val_indices)

        train_overall_losses = []
        train_state_losses = []
        train_reward_losses = []

        train_val_state_losses = []
        train_val_reward_losses = []

        val_state_losses = []
        val_reward_losses = []

        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        for epoch in range(epochs):
            overall_loss, state_loss, reward_loss = self.training_step(train_indices, epoch)
            train_overall_losses.append(overall_loss)
            train_state_losses.append(state_loss)
            train_reward_losses.append(reward_loss)

            #Evaluate with validation set for early stopping
            if epoch % self.eval_interval == 0:
                val_state_loss, val_reward_loss = self.validate(val_indices)
                val_state_losses.append(val_state_loss)
                val_reward_losses.append(val_reward_loss)
                train_val_state_loss, train_val_reward_loss = self.validate(train_indices)
                train_val_state_losses.append(train_val_state_loss)
                train_val_reward_losses.append(train_val_reward_loss)

                # change loss weighting
                weight_factors = np.ones(2)
                weights = np.array([train_val_state_loss * self.factor_state_loss, train_val_reward_loss * self.factor_reward_loss])
                for i in range(weights.shape[0]):
                    weight_factors[i] = weights[i] / np.sum(weights)
                self.loss_weight_state = weight_factors[0]
                self.loss_weight_reward = weight_factors[1]
                if int(os.environ['DEBUG']) == 1:
                    print("weight factors: " + str(weight_factors))

                # Debug printing
                if int(os.environ['DEBUG']) == 1:
                    print("\nEpoch: " + str(epoch))
                    print("Overall loss: " + str(train_overall_losses[-1]))
                    print("Train Validation loss (state, reward): " + str(train_val_state_losses[-1]) + ' , ' + str(train_val_reward_losses[-1]))
                    print("Validation loss (state, reward): " + str(val_state_losses[-1]) + ' , ' + str(val_reward_losses[-1]))
                if self.early_stopping(epoch, val_state_loss + val_reward_loss):
                    print("Early stopping at epoch " + str(epoch))
                    break

        # load the least loss encoder
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.decoder.load_state_dict(torch.load(self.decoder_path))
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

        # for logging
        validation_train = self.validate(train_indices)
        validation_val = self.validate(val_indices)

        logger.record_tabular("Reconstruction_train_val_state_loss", validation_train[0])
        logger.record_tabular("Reconstruction_train_val_reward_loss", validation_train[1])
        logger.record_tabular("Reconstruction_val_state_loss", validation_val[0])
        logger.record_tabular("Reconstruction_val_reward_loss", validation_val[1])
        logger.record_tabular("Reconstruction_epochs", epoch + 1)


    def training_step(self, indices, step):
        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder.
        The overall objective due to the generative model is:
        parameter* = arg max ELBO
        ELBO = sum_k q(y=k | x) * [ log p(x|z_k) - KL ( q(z, x,y=k) || p(z|y=k) ) ] - KL ( q(y|x) || p(y) )
        '''

        # get data from replay buffer
        # TODO: for validation data use all data --> batch size == validation size
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=True)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        # prepare for usage in decoder
        decoder_action = ptu.from_numpy(data['actions'])[:, -1, :]
        decoder_state = ptu.from_numpy(data['observations'])[:, -1, :]
        decoder_next_state = ptu.from_numpy(data['next_observations'])[:, -1, :]
        decoder_reward = ptu.from_numpy(data['rewards'])[:, -1, :]
        true_task = data['true_tasks'][:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (decoder_next_state - decoder_state)[:, :self.state_reconstruction_clip]
        else:
            decoder_state_target = decoder_next_state[:, :self.state_reconstruction_clip]

        # Forward pass through encoder
        y_distribution, z_distributions = self.encoder.encode(encoder_input)

        if self.isIndividualY:
            kl_qz_pz = ptu.zeros(self.batch_size, self.timesteps, self.num_classes)
            state_losses = ptu.zeros(self.batch_size, self.timesteps, self.num_classes)
            reward_losses = ptu.zeros(self.batch_size, self.timesteps, self.num_classes)
            nll_px = ptu.zeros(self.batch_size, self.timesteps, self.num_classes)

            # every y component (see ELBO formula)
            for y in range(self.num_classes):
                z, _ = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y)

                # put in decoder to get likelihood
                state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z)
                state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
                reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)
                state_losses[:, :, y] = state_loss.unsqueeze(1).repeat(1, self.timesteps)
                reward_losses[:, :, y] = reward_loss.unsqueeze(1).repeat(1, self.timesteps)
                nll_px[:, :, y] = self.loss_weight_state * state_losses[:, :, y] + self.loss_weight_reward * reward_losses[:, :, y]

                # KL ( q(z | x,y=k) || p(z|y=k) )
                prior = self.prior_pz(y)
                kl_qz_pz[:, :, y] = torch.sum(kl.kl_divergence(z_distributions[y], prior), dim=-1)

            # KL ( q(y | x) || p(y) )
            kl_qy_py = kl.kl_divergence(y_distribution, self.prior_py())

            # Overall ELBO
            if not self.component_constraint_learning:
                elbo = torch.sum(torch.sum(torch.mul(y_distribution.probs, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz), dim=-1) - self.beta_kl_y * kl_qy_py)

            # Component-constraint_learing
            if self.component_constraint_learning:
                temp = ptu.zeros_like(y_distribution.probs)
                index = ptu.from_numpy(np.array([a["base_task"] for a in true_task[:, 0].tolist()])).unsqueeze(1).expand(-1, self.timesteps).unsqueeze(2).long()
                true_task_multiplier = temp.scatter(2, index, 1)
                loss_ce = nn.CrossEntropyLoss(reduction='none')
                elbo = torch.sum(torch.sum(torch.mul(true_task_multiplier, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz), dim=-1) - loss_ce(y_distribution.probs, ptu.from_numpy(np.array([a["base_task"] for a in true_task[:, 0].tolist()])).unsqueeze(1).expand(-1, self.timesteps)))
            # but elbo should be maximized, and backward function assumes minimization
            loss = (-1) * elbo

            # Optimization strategy:
            # Decoder: the two head loss functions backpropagate their gradients into corresponding parts
            # of the network, then ONE common optimizer compute all weight updates
            # Encoder: the KLs and the likelihood from the decoder backpropagate their gradients into
            # corresponding parts of the network, then ONE common optimizer computes all weight updates
            # This is not done explicitly but all within the elbo loss.

            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer_decoder.step()
            self.optimizer_encoder.step()

            return ptu.get_numpy(loss) / self.batch_size, ptu.get_numpy(
                torch.sum(state_losses)) / self.batch_size, ptu.get_numpy(
                torch.sum(reward_losses)) / self.batch_size

        else:
            kl_qz_pz = ptu.zeros(self.batch_size, self.num_classes)
            state_losses = ptu.zeros(self.batch_size, self.num_classes)
            reward_losses = ptu.zeros(self.batch_size, self.num_classes)
            nll_px = ptu.zeros(self.batch_size, self.num_classes)

            # every y component (see ELBO formula)
            for y in range(self.num_classes):
                z, _ = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y)

                # put in decoder to get likelihood
                state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z)
                state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
                reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)
                state_losses[:, y] = state_loss
                reward_losses[:, y] = reward_loss
                nll_px[:, y] = self.loss_weight_state * state_loss + self.loss_weight_reward * reward_loss

                # KL ( q(z | x,y=k) || p(z|y=k) )
                prior = self.prior_pz(y)
                kl_qz_pz[:, y] = torch.sum(kl.kl_divergence(z_distributions[y], prior), dim=-1)

            # KL ( q(y | x) || p(y) )
            kl_qy_py = kl.kl_divergence(y_distribution, self.prior_py())

            # Overall ELBO
            if not self.component_constraint_learning:
                elbo = torch.sum(torch.sum(torch.mul(y_distribution.probs,  (-1) * nll_px - self.alpha_kl_z * kl_qz_pz), dim=-1) - self.beta_kl_y * kl_qy_py)

            # Component-constraint_learing
            if self.component_constraint_learning:
                temp = ptu.zeros_like(y_distribution.probs)
                true_task_multiplier = temp.scatter(1, ptu.from_numpy(np.array([a["base_task"] for a in true_task[:, 0].tolist()])).unsqueeze(1).long(), 1)
                loss_nll = nn.NLLLoss(reduction='none')
                target_label = ptu.from_numpy(np.array([a["base_task"] for a in true_task[:, 0].tolist()])).long()
                elbo = torch.sum(torch.sum(torch.mul(true_task_multiplier, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz), dim=-1) - self.beta_kl_y * loss_nll(torch.log(y_distribution.probs), target_label))
            # but elbo should be maximized, and backward function assumes minimization
            loss = (-1) * elbo

            # Optimization strategy:
            # Decoder: the two head loss functions backpropagate their gradients into corresponding parts
            # of the network, then ONE common optimizer compute all weight updates
            # Encoder: the KLs and the likelihood from the decoder backpropagate their gradients into
            # corresponding parts of the network, then ONE common optimizer computes all weight updates
            # This is not done explicitly but all within the elbo loss.

            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer_decoder.step()
            self.optimizer_encoder.step()

            return ptu.get_numpy(loss)/self.batch_size, ptu.get_numpy(torch.sum(state_losses, dim=0))/self.batch_size, ptu.get_numpy(torch.sum(reward_losses, dim=0))/self.batch_size

    def prior_pz(self, y):
        '''
        As proposed in the CURL paper: use linear layer, that conditioned on y gives Gaussian parameters
        OR
        Gaussian with N(y, 0.5)
        IF z not used:
        Just give back y with 0.01 variance
        '''

        if self.isIndividualY:
            if self.prior_mode == 'fixedOnY':
                return torch.distributions.normal.Normal(ptu.ones(self.batch_size, self.timesteps, 1) * y,
                                                         ptu.ones(self.batch_size, self.timesteps, 1) * self.prior_sigma)

            elif self.prior_mode == 'network':
                one_hot = ptu.zeros(self.batch_size, self.timesteps, self.num_classes)
                one_hot[:, :, y] = 1
                mu_sigma = self.prior_pz_layer(one_hot)#.detach()  # we do not want to backprop into prior
                return generate_gaussian(mu_sigma, self.latent_dim)
        else:
            if self.prior_mode == 'fixedOnY':
                return torch.distributions.normal.Normal(ptu.ones(self.batch_size, self.latent_dim) * y,
                                                         ptu.ones(self.batch_size, self.latent_dim) * self.prior_sigma)

            elif self.prior_mode == 'network':
                one_hot = ptu.zeros(self.batch_size, self.num_classes)
                one_hot[:, y] = 1
                mu_sigma = self.prior_pz_layer(one_hot)#.detach() # we do not want to backprop into prior
                return generate_gaussian(mu_sigma, self.latent_dim)

    def prior_py(self):
        '''
        Categorical uniform distribution
        '''
        if self.isIndividualY:
            return torch.distributions.categorical.Categorical(probs=ptu.ones(self.batch_size, self.timesteps, self.num_classes) * (1.0 / self.num_classes))
        else:
            return torch.distributions.categorical.Categorical(probs=ptu.ones(self.batch_size, self.num_classes) * (1.0 / self.num_classes))

    def validate(self, indices):
        # get data from replay buffer
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=True)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        # prepare for usage in decoder
        decoder_action = ptu.from_numpy(data['actions'])[:, -1, :]
        decoder_state = ptu.from_numpy(data['observations'])[:, -1, :]
        decoder_next_state = ptu.from_numpy(data['next_observations'])[:, -1, :]
        decoder_reward = ptu.from_numpy(data['rewards'])[:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (decoder_next_state - decoder_state)[:, :self.state_reconstruction_clip]
        else:
            decoder_state_target = decoder_next_state[:, :self.state_reconstruction_clip]

        z, y = self.encoder(encoder_input)
        state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z)
        state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
        reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)

        return ptu.get_numpy(torch.sum(state_loss)) / self.batch_size, ptu.get_numpy(torch.sum(reward_loss))/ self.batch_size

    def early_stopping(self, epoch, loss):
        if loss < self.lowest_loss:
            if int(os.environ['DEBUG']) == 1:
                print("Found new minimum at Epoch " + str(epoch))
            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch
            torch.save(self.encoder.state_dict(), self.encoder_path)
            torch.save(self.decoder.state_dict(), self.decoder_path)
        if epoch - self.lowest_loss_epoch > self.early_stopping_threshold:
            return True
        else:
            return False
