import torch
import torch.nn as nn
import numpy as np
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu

from cerml.scripted_policies import policies

class CEMRLAgent(nn.Module):
    def __init__(self,
                 encoder,
                 prior_pz,
                 policy
                 ):
        super(CEMRLAgent, self).__init__()
        self.encoder = encoder
        self.prior_pz = prior_pz
        self.policy = policy

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        state = ptu.from_numpy(state).view(1, -1)
        z, _ = self.encoder(encoder_input)
        if z_debug is not None:
            z = z_debug
        policy_input = torch.cat([state, z], dim=1)
        return self.policy.get_action(policy_input, deterministic=deterministic), np_ify(z.clone().detach())[0, :]

class ScriptedPolicyAgent(nn.Module):
    def __init__(self,
                 encoder,
                 prior_pz,
                 policy
                 ):
        super(ScriptedPolicyAgent, self).__init__()
        self.encoder = encoder
        self.prior_pz = prior_pz
        self.policy = policy
        self.latent_dim = encoder.latent_dim

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        env_name = env.active_env_name
        oracle_policy = policies[env_name]()
        action = oracle_policy.get_action(state)
        return (action.astype('float32'), {}), np.zeros(self.latent_dim, dtype='float32')
