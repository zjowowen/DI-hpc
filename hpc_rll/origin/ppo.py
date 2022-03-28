from collections import namedtuple
from typing import Optional, Tuple
import torch
from torch.distributions import Independent, Normal

ppo_data = namedtuple(
    'ppo_data', ['logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'adv', 'return_', 'weight']
)
ppo_continuous_data = namedtuple(
    'ppo_continuous_data', ['mu_new', 'sigma_new', 'mu_old', 'sigma_old', 'action', 'value_new', 'value_old', 'adv', 'return_', 'weight']
)
ppo_loss = namedtuple('ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])
ppo_info = namedtuple('ppo_info', ['approx_kl', 'clipfrac'])


def ppo_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
        dual_clip: Optional[float] = None
) -> Tuple[namedtuple, namedtuple]:
    """
        Overview:
            Implementation of Proximal Policy Optimization (arXiv:1707.06347) with value_clip and dual_clip
        Arguments:
            - data (:obj:`namedtuple`): the ppo input data with fieids shown in ``ppo_data``
            - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2
            - use_value_clip (:obj:`bool`): whether to use clip in value loss with the same ratio as policy
            - dual_clip (:obj:`float`): a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
            defaults to 5.0, if you don't want to use it, set this parameter to None
        Returns:
            - ppo_loss (:obj:`namedtuple`): the ppo loss item, all of them are the differentiable 0-dim tensor
            - ppo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar
        Shapes:
            - logit_new (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
            - logit_old (:obj:`torch.FloatTensor`): :math:`(B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(B, )`
            - value_new (:obj:`torch.FloatTensor`): :math:`(B, )`
            - value_old (:obj:`torch.FloatTensor`): :math:`(B, )`
            - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
            - return (:obj:`torch.FloatTensor`): :math:`(B, )`
            - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
            - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
            - value_loss (:obj:`torch.FloatTensor`): :math:`()`

        .. note::
            adv is already normalized value (adv - adv.mean()) / (adv.std() + 1e-8), and there are many
            ways to calculate this mean and std, like among data buffer or train batch, so we don't couple
            this part into ppo_error, you can refer to our examples for different ways.
    """
    assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
        dual_clip
    )
    logit_new, logit_old, action, value_new, value_old, adv, return_, weight = data
    if weight is None:
        weight = torch.ones_like(adv)
    dist_new = torch.distributions.categorical.Categorical(logits=logit_new)
    dist_old = torch.distributions.categorical.Categorical(logits=logit_old)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    entropy_loss = (dist_new.entropy() * weight).mean()
    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    if dual_clip is not None:
        policy_loss = (-torch.max(torch.min(surr1, surr2), dual_clip * adv) * weight).mean()
    else:
        policy_loss = (-torch.min(surr1, surr2) * weight).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    # value_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        value_loss = 0.5 * (torch.max(v1, v2) * weight).mean()
    else:
        value_loss = 0.5 * ((return_ - value_new).pow(2) * weight).mean()

    return ppo_loss(policy_loss, value_loss, entropy_loss), ppo_info(approx_kl, clipfrac)



def ppo_error_continuous(
    data: namedtuple,
    clip_ratio: float = 0.2,
    use_value_clip: bool = True,
    dual_clip: Optional[float] = None
) -> Tuple[namedtuple, namedtuple]:
    assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
        dual_clip
    )
    mu_new, sigma_new, mu_old, sigma_old, action, value_new, value_old, adv, return_, weight = data
    if weight is None:
        weight = torch.ones_like(adv)
    dist_new = Independent(Normal(mu_new, sigma_new), 1)
    if len(mu_old.shape) == 1:
        dist_old = Independent(Normal(mu_old.unsqueeze(-1), sigma_old.unsqueeze(-1)), 1)
    else:
        dist_old = Independent(Normal(mu_old, sigma_old), 1)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    entropy_loss = (dist_new.entropy() * weight).mean()
    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    if dual_clip is not None:
        policy_loss = (-torch.max(torch.min(surr1, surr2), dual_clip * adv) * weight).mean()
    else:
        policy_loss = (-torch.min(surr1, surr2) * weight).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    # value_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        value_loss = 0.5 * (torch.max(v1, v2) * weight).mean()
    else:
        value_loss = 0.5 * ((return_ - value_new).pow(2) * weight).mean()

    return ppo_loss(policy_loss, value_loss, entropy_loss), ppo_info(approx_kl, clipfrac)

