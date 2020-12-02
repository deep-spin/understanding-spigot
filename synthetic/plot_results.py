import sys
import json

from simple_latent_classifier import plot_results_range

plots = None #['Linear model', 'Gold labels', 'Softmax']
plot_file = 'results1.png'

if __name__ == '__main__':
    results_file = sys.argv[1]
    with open(results_file) as file:
        results = json.load(file)

    plot_file = results_file.replace('.txt', '.png')

    plot_results_range(results, plots, plot_file)

    print('Results written to: ', plot_file)
