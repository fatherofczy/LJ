import numpy as np
import matplotlib.pyplot as plt
from itertools import product

colorlist = ['y', 'm', 'c', 'g', 'b']
params = list(product(np.arange(2), np.arange(5)+1))

modelnames = {'Transformer-Vanilla-debug': 'Vanilla-Transformer'}
params = list(product(params, modelnames.keys()))
perf_list = []

for i, p in enumerate(params):
    perf = np.load(f'./Logfiles/{p[1]}-{p[0][0]}-{p[0][1]}.npz', allow_pickle=True)
    ret_q5 = int(np.nanmean(perf[f'GroupReturn5']) * 10000)
    ret_q1 = int(np.nanmean(perf[f'GroupReturn1']) * 10000)
    # group return
    for _ in range(5):
        plt.plot(np.nanmean(perf[f'GroupReturn{_+1}'], axis=0) * 10000, label=f'Group{_+1}', color=colorlist[_])
    plt.title(f'Model: {modelnames[p[1]]}\n epoch{p[0][0]+1}-{p[0][1]}\n q5: {ret_q5}, q1: {ret_q1}')
    plt.ylim(-50, 50)
    plt.legend(loc='upper left')
    plt.savefig(f'./Logfiles/{modelnames[p[1]]}-epoch{p[0][0]+1}-{p[0][1]}', dpi=200)
    plt.close()