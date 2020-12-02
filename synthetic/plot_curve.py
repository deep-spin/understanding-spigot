# collect grid search results

import sys
import json
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict


import numpy as np
from scipy.stats import sem

def fetch_curve(results_dir, tgt_lr, model_name):

    results = defaultdict(list)

    for file in os.listdir(results_dir):
        filename = os.fsdecode(file)
        if filename.endswith('.txt'):
            ixf = filename.index('features_')
            lr = filename[ixf:].split('_')[1]
            # lr = filename.split('_')[6]
            if lr not in ('best', str(tgt_lr)):
                continue

            with open(results_dir+'/'+filename) as file:
                content = json.load(file)

                for key, value in content.items():

                    if key.lower() == model_name:

                        # print(filename, key, lr)

                        for k in range(len(value['eval_accuracies'])):
                            for t in range(len(value['eval_accuracies'][k])):
                                acc = 100 * value['eval_accuracies'][k][t]
                                vm = 100 * value['eval_vmeasures'][k][t]
                                tacc = 100 * value['test_accuracies'][k][t]
                                tvm = 100 * value['test_vmeasures'][k][t]

                                # results['model_name'].append(key)
                                results['t'].append(t)
                                results['valid_acc'].append(acc)
                                results['valid_vm'].append(vm)
                                results['test_acc'].append(tacc)
                                results['test_vm'].append(tvm)
    df = pd.DataFrame(results)
    df = (df
          .groupby(['t'])
          .agg(['median', sem]))
#          .sort_values(('valid_acc', 'median'))
#          .groupby('gr')
#          .last())
    return df


if __name__ == '__main__':

    n_clusters = 10

    plot_args = {
        'ste-i': dict(label="STE-I", marker='.', ls=":"),
        'spigot': dict(label="SPIGOT", marker='o'),
        'spigot-crossentr-argmaxfwd': dict(label="SPIGOT-CE", marker='s'),
        'linear model': dict(label="Linear", marker='s'),
        'gold labels': dict(label="Gold clusters", marker='s'),
        'softmax-relaxedtest': dict(label="Softmax", marker='s'),
        # 'spigot-expgrad-argmaxfwd': dict(label="SPIGOT-EG", marker='x')
    }

    if n_clusters == 3:
        results_dir = "results"
        model_lrs = [
            ('linear model', 0.001),
            ('gold labels', 0.001),
            ('softmax-relaxedtest-st1-gr1', 0.002),
            ('sparsemax-relaxedtest-st1-gr1', 0.001),
            ('gumbel softmax-relaxedtest-st1-gr1', 0.002),
            ('gumbel ste-st1-gr1', 0.002),
            ('reinforce no baseline', 0.002),
            ('reinforce with baseline', 0.002),
            ('ste-s-st1-gr1', 0.002),
            ('ste-i-st2-gr1', 0.001),
            ('spigot-st1-gr1', 0.001),
            ('spigot-crossentr-argmaxfwd-st2-gr1', 0.002),
            ('spigot-expgrad-argmaxfwd-st0.1-gr1', 0.002),
        ]

    if n_clusters == 10:
        results_dir = "./final_results_n/results_10"
        model_lrs = [
            ('linear model', 0.002),
            ('gold labels', 0.001),
            ('softmax-relaxedtest-st1-gr1', 0.001),
            # ('sparsemax-relaxedtest-st1-gr1', 0.002),
            # ('gumbel softmax-relaxedtest-st1-gr1', 0.002),
            # ('gumbel ste-st1-gr1', 0.002),
            # ('reinforce no baseline', 0.001),
            # ('reinforce with baseline', 0.002),
            # ('ste-s-st1-gr1', 0.001),
            ('ste-i-st1-gr1', 0.002),
            ('spigot-st0.1-gr1', 0.0001),
            ('spigot-crossentr-argmaxfwd-st1-gr1', 0.002),
            # ('spigot-expgrad-argmaxfwd-st1-gr1', 0.002),
        ]

    all_res = []
    fig, (ax_acc, ax_vm) = plt.subplots(1, 2, constrained_layout=True,
                                        figsize=(9, 3))
    for model_name, lr in model_lrs:

        results = fetch_curve(results_dir, lr, model_name)
        model_name = model_name.split('-st')[0]
        results.plot(y=('valid_acc', 'median'),
                     yerr=results['valid_acc', 'sem'],
                     ax=ax_acc, legend=False,
                     capsize=2, elinewidth=2, linewidth=2,
                     **plot_args[model_name])
        results.plot(y=('valid_vm', 'median'),
                     yerr=results['valid_vm', 'sem'],
                     ax=ax_vm, legend=False,
                     capsize=2, elinewidth=2, linewidth=2,
                     **plot_args[model_name])
    # ax_acc.set_ylim(35, None)
    ax_vm.legend(ncol=2)
    # ax_acc.legend(ncol=2)
    ax_acc.set_ylabel('valid. accuracy')
    ax_vm.set_ylabel('valid. v-measure')
    ax_acc.set_xlabel('epoch (CHECK)')
    ax_vm.set_xlabel('epoch (CHECK)')
    plt.savefig("curve.pdf")
    # plt.show()
