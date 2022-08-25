import time
import torch
import torch.nn.functional as F
from hpc_rll.origin.ppo import ppo_error, ppo_data, ppo_error_continuous, ppo_continuous_data
from hpc_rll.rl_utils.ppo import PPO, PPOContinuous
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

B = 128
N = 128
clip_ratio = 0.2
use_value_clip = True
dual_clip = None

warm_up_times = 100

def generate_data_ppo():
    ori_logits_new = torch.randn(B, N)
    ori_logits_old = torch.randn(B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_value_new = torch.randn(B)
    ori_value_old = torch.randn(B)
    ori_adv = torch.randn(B)
    ori_return = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_logits_new = ori_logits_new.clone().detach()
    hpc_logits_old = ori_logits_old.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_value_new = ori_value_new.clone().detach()
    hpc_value_old = ori_value_old.clone().detach()
    hpc_adv = ori_adv.clone().detach()
    hpc_return = ori_return.clone().detach()
    hpc_weight = ori_weight.clone().detach()

    if use_cuda:
        ori_logits_new = ori_logits_new.cuda()
        ori_logits_old = ori_logits_old.cuda()
        ori_action = ori_action.cuda()
        ori_value_new = ori_value_new.cuda()
        ori_value_old = ori_value_old.cuda()
        ori_adv = ori_adv.cuda()
        ori_return = ori_return.cuda()
        ori_weight = ori_weight.cuda()

        hpc_logits_new = hpc_logits_new.cuda()
        hpc_logits_old = hpc_logits_old.cuda()
        hpc_action = hpc_action.cuda()
        hpc_value_new = hpc_value_new.cuda()
        hpc_value_old = hpc_value_old.cuda()
        hpc_adv = hpc_adv.cuda()
        hpc_return = hpc_return.cuda()
        hpc_weight = hpc_weight.cuda()

    ori_logits_new.requires_grad_(True)
    ori_value_new.requires_grad_(True)

    hpc_logits_new.requires_grad_(True)
    hpc_value_new.requires_grad_(True)

    return ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
        hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight

def ppo_val():
    hpc_ppo = PPO(B, N)

    if use_cuda:
        hpc_ppo = hpc_ppo.cuda()

    ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
        hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = generate_data_ppo()

    ori_loss, ori_info = ppo_error(ppo_data(ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight), clip_ratio, use_value_clip, dual_clip)
    ori_loss = sum(ori_loss)
    ori_loss.backward()

    hpc_loss, hpc_info = hpc_ppo(hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight, clip_ratio, use_value_clip, dual_clip)
    hpc_loss = sum(hpc_loss)
    hpc_loss.backward()

    print("ori_info: " + str(ori_info))
    print("hpc_info: " + str(hpc_info))
    mre = mean_relative_error(torch.flatten(ori_loss).cpu().detach().numpy(), torch.flatten(hpc_loss).cpu().detach().numpy())
    print("ppo fp loss mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_logits_new.grad).cpu().detach().numpy(), torch.flatten(hpc_logits_new.grad).cpu().detach().numpy())
    print("ppo bp logits_new mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_value_new.grad).cpu().detach().numpy(), torch.flatten(hpc_value_new.grad).cpu().detach().numpy())
    print("ppo bp value_new mean_relative_error: " + str(mre))

def ppo_perf(do_backward):

    hpc_ppo = PPO(B, N)

    if use_cuda:
        hpc_ppo = hpc_ppo.cuda()

    ori_test_data = []
    hpc_test_data = []
    for i in range(times):
        ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
            hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = generate_data_ppo()

        ori_test_data.append((
            ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight,
        ))
        hpc_test_data.append((
            hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight,
        ))

    ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
        hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = generate_data_ppo()

    for i in range(warm_up_times):
        ori_loss, ori_info = ppo_error(ppo_data(ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight), clip_ratio, use_value_clip, dual_clip)
        ori_loss = sum(ori_loss)
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight = ori_test_data[i]
        t0 = time.time()
        ori_loss, ori_info = ppo_error(ppo_data(ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight), clip_ratio, use_value_clip, dual_clip)
        
        if do_backward:
            ori_loss = sum(ori_loss)
            ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, origin ppo cost time: {}'.format(i, t1 - t0))

    for i in range(warm_up_times):
        hpc_loss, hpc_info = hpc_ppo(hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight, clip_ratio, use_value_clip, dual_clip)
        hpc_loss = sum(hpc_loss)
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = hpc_test_data[i]
        t0 = time.time()
        hpc_loss, hpc_info = hpc_ppo(hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight, clip_ratio, use_value_clip, dual_clip)
        
        if do_backward:
            hpc_loss = sum(hpc_loss)
            hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, hpc ppo cost time: {}'.format(i, t1 - t0))

def generate_data_ppo_continuous():
    ori_mu_new = torch.rand(B, N)
    ori_sigma_new = torch.rand(B, N)
    ori_mu_old = ori_mu_new + torch.rand_like(ori_mu_new) * 0.1
    ori_sigma_old = ori_sigma_new + torch.rand_like(ori_sigma_new) * 0.1
    ori_action = torch.randint(0, N, size=(B, 1))
    ori_value_new = torch.randn(B)
    ori_value_old = ori_value_new + torch.rand_like(ori_value_new) * 0.1
    ori_adv = torch.rand(B)
    ori_return = torch.randn(B) * 2
    ori_weight = torch.randn(B)

    hpc_mu_new = ori_mu_new.clone().detach()
    hpc_sigma_new = ori_sigma_new.clone().detach()
    hpc_mu_old = ori_mu_old.clone().detach()
    hpc_sigma_old = ori_sigma_old.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_value_new = ori_value_new.clone().detach()
    hpc_value_old = ori_value_old.clone().detach()
    hpc_adv = ori_adv.clone().detach()
    hpc_return = ori_return.clone().detach()
    hpc_weight = ori_weight.clone().detach()

    if use_cuda:
        ori_mu_new = ori_mu_new.cuda()
        ori_sigma_new = ori_sigma_new.cuda()
        ori_mu_old = ori_mu_old.cuda()
        ori_sigma_old = ori_sigma_old.cuda()
        ori_action = ori_action.cuda()
        ori_value_new = ori_value_new.cuda()
        ori_value_old = ori_value_old.cuda()
        ori_adv = ori_adv.cuda()
        ori_return = ori_return.cuda()
        ori_weight = ori_weight.cuda()

        hpc_mu_new = hpc_mu_new.cuda()
        hpc_sigma_new = hpc_sigma_new.cuda()
        hpc_mu_old = hpc_mu_old.cuda()
        hpc_sigma_old = hpc_sigma_old.cuda()
        hpc_action = hpc_action.cuda()
        hpc_value_new = hpc_value_new.cuda()
        hpc_value_old = hpc_value_old.cuda()
        hpc_adv = hpc_adv.cuda()
        hpc_return = hpc_return.cuda()
        hpc_weight = hpc_weight.cuda()

    ori_mu_new.requires_grad_(True)
    ori_sigma_new.requires_grad_(True)
    ori_value_new.requires_grad_(True)

    hpc_mu_new.requires_grad_(True)
    hpc_sigma_new.requires_grad_(True)
    hpc_value_new.requires_grad_(True)

    return ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
        hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight

def ppo_continuous_val():

    hpc_ppo_continuous = PPOContinuous(B, N)

    if use_cuda:
        hpc_ppo_continuous = hpc_ppo_continuous.cuda()

    ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
        hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = generate_data_ppo_continuous()

    ori_loss, ori_info = ppo_error_continuous(ppo_continuous_data(ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight), clip_ratio, use_value_clip, dual_clip)
    ori_loss = sum(ori_loss)
    ori_loss.backward()


    hpc_loss, hpc_info = hpc_ppo_continuous(hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight, clip_ratio, use_value_clip, dual_clip)
    hpc_loss = sum(hpc_loss)
    hpc_loss.backward()

    print("ori_info: " + str(ori_info))
    print("hpc_info: " + str(hpc_info))
    mre = mean_relative_error(torch.flatten(ori_loss).cpu().detach().numpy(), torch.flatten(hpc_loss).cpu().detach().numpy())
    print("ppo fp loss mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_mu_new.grad).cpu().detach().numpy(), torch.flatten(hpc_mu_new.grad).cpu().detach().numpy())
    print("ppo bp mu_new mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_sigma_new.grad).cpu().detach().numpy(), torch.flatten(hpc_sigma_new.grad).cpu().detach().numpy())
    print("ppo bp sigma_new mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_value_new.grad).cpu().detach().numpy(), torch.flatten(hpc_value_new.grad).cpu().detach().numpy())
    print("ppo bp value_new mean_relative_error: " + str(mre))

def ppo_continuous_perf(do_backward):

    hpc_ppo_continuous = PPOContinuous(B, N)

    if use_cuda:
        hpc_ppo_continuous = hpc_ppo_continuous.cuda()

    ori_test_data = []
    hpc_test_data = []
    for i in range(times):
        ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
            hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = generate_data_ppo_continuous()

        ori_test_data.append((
            ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight,
        ))
        hpc_test_data.append((
            hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight,
        ))

    ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight, \
        hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = generate_data_ppo_continuous()

    for i in range(warm_up_times):
        ori_loss, ori_info = ppo_error_continuous(ppo_continuous_data(ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight), clip_ratio, use_value_clip, dual_clip)
        ori_loss = sum(ori_loss)
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight = ori_test_data[i]
        t0 = time.time()
        ori_loss, ori_info = ppo_error_continuous(ppo_continuous_data(ori_mu_new, ori_sigma_new, ori_mu_old, ori_sigma_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight), clip_ratio, use_value_clip, dual_clip)
        if do_backward:
            ori_loss = sum(ori_loss)
            ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, origin ppo continuous cost time: {}'.format(i, t1 - t0))

    for i in range(warm_up_times):
        hpc_loss, hpc_info = hpc_ppo_continuous(hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight, clip_ratio, use_value_clip, dual_clip)
        hpc_loss = sum(hpc_loss)
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight = hpc_test_data[i]
        t0 = time.time()
        hpc_loss, hpc_info = hpc_ppo_continuous(hpc_mu_new, hpc_sigma_new, hpc_mu_old, hpc_sigma_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight, clip_ratio, use_value_clip, dual_clip)
        if do_backward:
            hpc_loss = sum(hpc_loss)
            hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, hpc ppo continuous cost time: {}'.format(i, t1 - t0))

if __name__ == '__main__':
    print("target problem: B = {}, N = {}, clip_ratio = {}, use_value_clip = {}, dual_clip = {}".format(B, N, clip_ratio, use_value_clip, dual_clip))

    print("================run ppo validation test================")
    ppo_val()
    print("================run ppo performance test================")
    print("----------------run ppo forward only----------------")
    ppo_perf(do_backward=False)
    print("----------------run ppo forward and backward----------------")
    ppo_perf(do_backward=True)

    print("================run ppo continuous validation test================")
    ppo_continuous_val()
    print("================run ppo continuous performance test================")
    print("----------------run ppo continuous forward only----------------")
    ppo_continuous_perf(do_backward=False)
    print("----------------run ppo continuous forward and backward----------------")
    ppo_continuous_perf(do_backward=True)