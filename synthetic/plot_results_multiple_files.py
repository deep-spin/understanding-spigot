import sys
import json
import os
import re

from main import plot_results_range

plots = None #['Linear model', 'Gold labels', 'Softmax']
# plot_file = 'results1.png'

relaxed = ['Linear', 'Gold labels', 'Softmax-RelaxedTest', 'Sparsemax-RelaxedTest',\
             'SPIGOT-CrossEntr-RelaxedTest', 'SPIGOT-ExpGrad-RelaxedTest',
             'Gumbel Softmax-RelaxedTest']
choice = ['SPIGOT-CrossEntr-ArgmaxTest', 'SPIGOT-ExpGrad-ArgmaxTest', \
            'STE-I', 'STE-S', 'SPIGOT', 'Gumbel Softmax-ArgmaxTest', 'Gumbel STE', \
            'Softmax-ArgmaxTest', 'Sparsemax-ArgmaxTest', \
            'REINFORCE No Baseline', 'REINFORCE With Baseline',
            'Linear model', 'Gold labels']

choice_train = ['STE-I', 'STE-S', 'SPIGOT', 'Gumbel STE', \
            'REINFORCE No Baseline', 'REINFORCE With Baseline']
            

if __name__ == '__main__':
    _, results_dir, file_name_match, model_name_match = sys.argv

    # model_name_match = '.*'

    file_pattern = re.compile(file_name_match)
    model_pattern = re.compile(model_name_match)

    results = {}
    for file in os.listdir(results_dir):
        filename = os.fsdecode(file)
        # if file_name_match in filename and filename.endswith('.txt'):
        if file_pattern.search(filename) and filename.endswith('.txt'):
        # if re.search(file_name_match+'*.txt', filename):
            print('file match:', filename)

            with open(results_dir+'/'+filename) as file:
                content = json.load(file)

                for key, value in content.items():
                    key = key.replace('-st1-gr1', '')
                    print('key:', key)
                    if model_pattern.search(key.lower()):
                    # if key in choice:
                        print('content match:', key)
                        results[key] = value


    plot_file = f"{results_dir}/_temp_{model_name_match}_{file_name_match}.png"

    plot_results_range(results, plots, plot_file)

    print('Results written to: ', plot_file)
