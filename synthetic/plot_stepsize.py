
# collect grid search results

import sys
import json
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib

from collections import defaultdict


import numpy as np
from scipy.stats import sem


def fetch_combined(results_dir, tgt_lr, model_name):

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
                    try:
                        method, st, gr = key.rsplit('-', 2)
                    except:
                        continue

                    if gr != 'gr1':
                        continue

                    if method.lower() == model_name:

                        # print(filename, key, lr)

                        for k in range(len(value['eval_accuracies'])):
                            acc = 100 * value['eval_accuracies'][k][-1]
                            vm = 100 * value['eval_vmeasures'][k][-1]
                            tacc = 100 * value['test_accuracies'][k][-1]
                            tvm = 100 * value['test_vmeasures'][k][-1]

                            results['st'].append(float(st[2:]))
                            results['valid_acc'].append(acc)
                            results['valid_vm'].append(vm)
                            results['test_acc'].append(tacc)
                            results['test_vm'].append(tvm)
    df = pd.DataFrame(results).groupby(['st'])
    df = (df
          .agg(['median', sem]))
    print(df)
    return df


if __name__ == '__main__':
    n_clusters = 10

    plot_args = {
        'ste-i': dict(label="STE-I", marker='.', ls=":"),
        'spigot': dict(label="SPIGOT", marker='o'),
        'spigot-crossentr-argmaxfwd': dict(label="SPIGOT-CE", marker='s'),
        'spigot-expgrad-argmaxfwd': dict(label="SPIGOT-EG", marker='x')
    }

    if n_clusters == 3:
        results_dir = "results"
        model_lrs = [
            ('ste-i', 0.001),
            ('spigot', 0.001),
            ('spigot-crossentr-argmaxfwd', 0.002),
            ('spigot-expgrad-argmaxfwd', 0.002),
        ]

    if n_clusters == 10:
        results_dir = "results_10"
        model_lrs = [
            ('ste-i',  0.002),
            ('spigot', 0.0001),
            ('spigot-crossentr-argmaxfwd', 0.002),
            ('spigot-expgrad-argmaxfwd', 0.002),
        ]

    fig, (ax_acc, ax_vm) = plt.subplots(1, 2, constrained_layout=True,
                                        figsize=(4.2,2.5))

    for model_name, lr in model_lrs:
        results = fetch_combined(results_dir, lr, model_name)

        results['valid_acc', 'median'].plot(yerr=results['valid_acc', 'sem'],
                                            ax=ax_acc,
                                            capsize=2, elinewidth=2, linewidth=2,
                                            **plot_args[model_name])
        results['valid_vm', 'median'].plot(yerr=results['valid_vm', 'sem'],
                                           capsize=2, elinewidth=2, linewidth=2,
                                           ax=ax_vm,
                                           **plot_args[model_name])
    # ax_acc.semilogx()
    # ax_vm.semilogx()
    ax_acc.set_xlim(0, 2.1)
    ax_vm.set_xlim(0, 2.1)
    ax_vm.set_xticks((.1, 1, 2))
    ax_acc.set_xticks((.1, 1, 2))
    ax_acc.set_xlabel("pull-back step size")
    ax_vm.set_xlabel("pull-back step size")
    ax_acc.set_ylabel("valid. accuracy")
    ax_vm.set_ylabel("valid. v-measure")
    ax_vm.legend()
    # plt.show()
    plt.savefig("impact_st.pdf")
    # plt.show()
    # tikzplotlib.save('impact_gr.tex')

