import numpy as np
import torch

from cerml.stacked_replay_buffer import StackedReplayBuffer
from cerml.encoder_decoder_networks import EncoderMixtureModelTrajectory

import rlkit.torch.pytorch_util as ptu


class Relabeler:
    def __init__(self,
                 encoder: EncoderMixtureModelTrajectory,
                 replay_buffer: StackedReplayBuffer,
                 batch_size_relabel,
                 action_space,
                 observation_space,
                 normalize
                 ):
        self.encoder = encoder
        self.replay_buffer = replay_buffer
        self.batch_size_relabel = batch_size_relabel
        self.action_space = action_space
        self.observation_space = observation_space

        self.normalize = normalize

    def relabel(self):
        all_points = self.replay_buffer.get_allowed_points()
        total_points = len(all_points)
        start = 0
        while start < total_points:
            if total_points - start < self.batch_size_relabel:
                batch_size = total_points - start
            else:
                batch_size = self.batch_size_relabel
            points = all_points[start:start + batch_size]
            data = self.replay_buffer.sample_few_step_batch(points, batch_size, normalize=self.normalize)
            encoder_input = self.replay_buffer.make_encoder_data(data, batch_size)
            z, y = self.encoder(encoder_input)

            # Uncomment to compute last next_z exactly.
            # Right now we use the last z as next_z, as next_z is actually never used
            '''
            # a lot to do for computing next_z (as there is no more future data point)
            observations = ptu.from_numpy(data['observations'])[-1,1:,:]
            actions = ptu.from_numpy(data['actions'])[-1,1:,:]
            rewards = ptu.from_numpy(data['rewards'])[-1,1:,:]
            next_observations = ptu.from_numpy(data['next_observations'])[-1,1:,:]

            observations = torch.cat([observations, ptu.from_numpy(data['next_observations'])[-1,-1,:].view(1,-1)])
            actions = torch.cat([actions, ptu.zeros(1, self.action_space)])
            rewards = torch.cat([rewards, ptu.zeros(1, 1)])
            next_observations = torch.cat([next_observations, ptu.zeros(1, self.observation_space)])

            encoder_input = torch.cat([observations, actions, rewards, next_observations], dim=1)
            mode = 'multiply'
            if mode == 'fully':
                # size: [batch_size, time_steps * (obs+action+reward)]
                encoder_input = encoder_input.view(1, -1)
            if mode == 'multiply':
                encoder_input.unsqueeze_(0)

            last_z = self.encoder(encoder_input)
            '''

            next_z = torch.cat([z[1:], z[-1].unsqueeze(0)])

            self.replay_buffer.relabel_z(start, batch_size, ptu.get_numpy(z), ptu.get_numpy(next_z), ptu.get_numpy(y))

            start += batch_size


