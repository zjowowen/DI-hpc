import time
import torch
from hpc_rll.origin.gru import GRUGatingUnit
from hpc_rll.rl_utils.gru import GRU
from testbase import mean_relative_error, times

T, B, input_dim = 1, 2, 4
#T, B, input_dim = 4, 12, 32

use_cuda = True
warm_up_times = 100


def generate_data():
    ori_x = torch.rand((T, B, input_dim))
    ori_y = torch.rand((T, B, input_dim))

    hpc_x = ori_x.clone().detach()
    hpc_y = ori_y.clone().detach()

    if use_cuda:
        ori_x = ori_x.cuda()
        ori_y = ori_y.cuda()

        hpc_x = hpc_x.cuda()
        hpc_y = hpc_y.cuda()

    return ori_x, ori_y, hpc_x, hpc_y


def gru_val():
    pass


def gru_perf():

    ori_gru = GRUGatingUnit(input_dim, 1.)
    hpc_gru = GRU(T, B, input_dim=input_dim, bg=1.)

    if use_cuda:
        ori_gru.cuda()
        hpc_gru.cuda()

    ori_test_data = []
    hpc_test_data = []
    for i in range(times):
        ori_x, ori_y, hpc_x, hpc_y = generate_data()
        ori_test_data.append((
            ori_x,
            ori_y,
        ))
        hpc_test_data.append((
            hpc_x,
            hpc_y,
        ))

    ori_x, ori_y, hpc_x, hpc_y = generate_data()

    for i in range(warm_up_times):
        ori_out = ori_gru(ori_x, ori_y)
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        ori_x, ori_y = ori_test_data[i]
        t0 = time.time()
        ori_out = ori_gru(ori_x, ori_y)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, origin coma cost time: {}'.format(i, t1 - t0))

    for i in range(warm_up_times):
        hpc_out = hpc_gru(hpc_x, hpc_y)
        if use_cuda:
            torch.cuda.synchronize()

    for i in range(times):
        hpc_x, hpc_y = hpc_test_data[i]
        t0 = time.time()
        hpc_out = hpc_gru(hpc_x, hpc_y)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        print('epoch: {}, hpc coma cost time: {}'.format(i, t1 - t0))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}, input_dim = {}".format(
        T, B, input_dim))
    print("================run gru validation test================")
    gru_val()
    print("================run gru performance test================")
    gru_perf()
