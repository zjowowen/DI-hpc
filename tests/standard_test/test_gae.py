import time
import torch
from hpc_rll.origin.gae import gae, gae_data
from hpc_rll.rl_utils.gae import GAE
from testbase import mean_relative_error, times

T = 1024
B = 64

use_cuda = True
warm_up_times = 100


def generate_data():

    ori_value = torch.randn(T + 1, B)
    ori_reward = torch.randn(T, B)

    hpc_value = ori_value.clone().detach()
    hpc_reward = ori_reward.clone().detach()

    if use_cuda:
        ori_value = ori_value.cuda()
        ori_reward = ori_reward.cuda()

        hpc_value = hpc_value.cuda()
        hpc_reward = hpc_reward.cuda()

    return ori_value, ori_reward, hpc_value, hpc_reward


def gae_val():

    ori_value, ori_reward, hpc_value, hpc_reward = generate_data()

    ori_gae = gae
    hpc_gae = GAE(T, B)
    if use_cuda:
        hpc_gae = hpc_gae.cuda()
    
    ori_adv = ori_gae(gae_data(ori_value, ori_reward))
    hpc_adv = hpc_gae(hpc_value, hpc_reward)
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(torch.flatten(ori_adv).cpu().detach().numpy(), torch.flatten(hpc_adv).cpu().detach().numpy())
    print("gae mean_relative_error: " + str(mre))

def gae_perf():

    ori_gae = gae
    hpc_gae = GAE(T, B)
    if use_cuda:
        hpc_gae = hpc_gae.cuda()

    ori_test_data = []
    hpc_test_data = []
    for i in range(times):
        ori_value, ori_reward, hpc_value, hpc_reward = generate_data()
        ori_test_data.append((
            ori_value,
            ori_reward,
        ))
        hpc_test_data.append((
            hpc_value,
            hpc_reward,
        ))

    ori_value, ori_reward, hpc_value, hpc_reward = generate_data()

    for i in range(warm_up_times):
        ori_adv = ori_gae(gae_data(ori_value, ori_reward))
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        ori_value, ori_reward = ori_test_data[i]
        t0 = time.time()
        ori_adv = ori_gae(gae_data(ori_value, ori_reward))
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, origin gae cost time: {}'.format(i, t1 - t0))

    for i in range(warm_up_times):
        hpc_adv = hpc_gae(hpc_value, hpc_reward)
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        hpc_value, hpc_reward = hpc_test_data[i]
        t0 = time.time()
        hpc_adv = hpc_gae(hpc_value, hpc_reward)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, hpc gae cost time: {}'.format(i, t1 - t0))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}".format(T, B))
    print("================run gae validation test================")
    gae_val()
    print("================run gae performance test================")
    gae_perf()
