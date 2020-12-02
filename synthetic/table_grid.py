# collect grid search results

import sys
import json
import os
import re

import pandas as pd

from collections import defaultdict


import numpy as np
from scipy.stats import sem

def _mean_std(x):
    #mean = np.mean(x)
    mean = np.median(x)
    # std = np.std(x)
    std = sem(x)

    return f"{100*mean:.2f}&{100*std:.2f}"


def fetch_table(results_dir, tgt_lr, model_name):

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
                            acc = 100 * value['eval_accuracies'][k][-1]
                            vm = 100 * value['eval_vmeasures'][k][-1]
                            tacc = 100 * value['test_accuracies'][k][-1]
                            tvm = 100 * value['test_vmeasures'][k][-1]

                            results['model_name'].append(key)
                            results['valid_acc'].append(acc)
                            results['valid_vm'].append(vm)
                            results['test_acc'].append(tacc)
                            results['test_vm'].append(tvm)
    df = pd.DataFrame(results)
    df = (df
          .groupby(['model_name'])
          .agg(['median', sem]))
#          .sort_values(('valid_acc', 'median'))
#          .groupby('gr')
#          .last())
    return df


def fetch_combined(results_dir, tgt_lr, model_name):
    # _, results_dir, lr, model_name = sys.argv
    # file_pattern = re.compile(file_name_match)
    # model_pattern = re.compile(model_name_match)


    results = defaultdict(list)


    for file in os.listdir(results_dir):
        filename = os.fsdecode(file)
        if filename.endswith('.txt'):
            ixf = filename.index('features_')
            lr = filename[ixf:].split('_')[1]
            print(lr)
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

                    if method.lower() == model_name:

                        print(filename, key, lr)

                        for k in range(len(value['eval_accuracies'])):
                            acc = value['eval_accuracies'][k][-1]
                            vm = value['eval_vmeasures'][k][-1]
                            tacc = value['test_accuracies'][k][-1]
                            tvm = value['test_vmeasures'][k][-1]

                            results['st'].append(st)
                            results['gr'].append(gr)
                            results['valid_acc'].append(acc)
                            results['valid_vm'].append(vm)
                            results['test_acc'].append(tacc)
                            results['test_vm'].append(tvm)
    df = pd.DataFrame(results).groupby(['gr', 'st'])
    print(df.count())
    df = (df
          .agg(['median', sem])
          .sort_values(('valid_acc', 'median'))
          .groupby('gr')
          .last())
    print(df)
    return df
    # print(len(valid_accs), valid_accs)
    # print(model_name_match, "\n&", _mean_std(test_accs), "&", _mean_std(test_vm),
            # r"\\")
    # print("valid acc", _mean_std(valid_accs))
    # print("valid  vm", _mean_std(valid_vm))
    # print(" test acc", _mean_std(test_accs))
    # print(" test  vm", _mean_std(test_vm))


if __name__ == '__main__':

    n_clusters = 3

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

        all_res = []
        for model_name, lr in model_lrs:
            results = fetch_table(results_dir, lr, model_name)
            all_res.append(results)


    if n_clusters == 10:
        results_dir = "results_10"
        model_lrs = [
            ('linear model', 0.002),
            ('gold labels', 0.001),
            ('softmax-relaxedtest-st1-gr1', 0.001),
            ('sparsemax-relaxedtest-st1-gr1', 0.002),
            ('gumbel softmax-relaxedtest-st1-gr1', 0.002),
            ('gumbel ste-st1-gr1', 0.002),
            ('reinforce no baseline', 0.001),
            ('reinforce with baseline', 0.002),
            ('ste-s-st1-gr1', 0.001),
            ('ste-i-st1-gr1', 0.002),
            ('spigot-st0.1-gr1', 0.0001),
            ('spigot-crossentr-argmaxfwd-st1-gr1', 0.002),
            ('spigot-expgrad-argmaxfwd-st1-gr1', 0.002),
        ]

        all_res = []
        for model_name, lr in model_lrs:
            results = fetch_table(results_dir, lr, model_name)
            all_res.append(results)

        # model_lrs = [
            # # ('spigot', 0.0001),
            # # ('ste-i', 0.002),
            # # ('spigot-crossentr-argmaxfwd', 0.002),
            # # ('spigot-expgrad-argmaxfwd', 0.002),
            # ('ste-s', 0.001)
        # ]

        # for model_name, lr in model_lrs:
            # results = fetch_combined(results_dir, lr, model_name)

    result = pd.concat(all_res)
    # print(result)

    # print(result.to_latex(columns=['test_acc', 'test_vm']))
    # print(result[['test_acc', 'test_vm']].to_latex(float_format="%%.2f"))
    print(result[['test_acc', 'test_vm']].to_latex(float_format=lambda x: "{:.2f}".format(x)))

    exit()
