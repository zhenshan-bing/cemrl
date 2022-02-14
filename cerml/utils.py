import torch
import torch.nn.functional as F


def product_of_gaussians3D(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared

def generate_gaussian(mu_sigma, latent_dim, sigma_ops="softplus", mode=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    if mode == 'multiplication':
        mu, sigma = product_of_gaussians3D(mus, sigmas)
    else:
        mu = mus
        sigma = sigmas
    return torch.distributions.normal.Normal(mu, sigma)


