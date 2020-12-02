import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

from models import *
from utils import RunningStats
from generate_latent_clf import make_latent_triples
from ste import _one_hot_argmax #, step_size, gradient_update_steps

import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_function = torch.nn.CrossEntropyLoss()

DEMO = False
save_model = False

# Number of runs
n_runs = 1

learning_rate = 0.002 # 0.0001, 0.001, 0.002
# learning_rate = "best"
BEST_LRS = {
    'LinearClassifier': 0.001,
    'GoldLatentClassifier': 0.001,
    'LatentSoftmax': 0.002,
    'LatentSparsemax': 0.001,
    'LatentSTESoftmax': 0.002,
    'LatentGumbelSoftmax': 0.002,
    'LatentGumbelSTE': 0.002,
    'REINFORCEClassifier' : 0.002,
    'LatentSTEIdentity': 0.001,
    'LatentSPIGOT': 0.001,
    'LatentSPIGOTCEArgmax': 0.002,
    'LatentSPIGOTEGArgmax': 0.002,
}

run_baselines = False # LR grid search - set it to True
run_reinforce = True # LR grid search - set it to True
run_latent_model = True
run_exact = False
run_minimum_risk_training = False

# step_size_grid_search = [0.1, 1, 2]
# grad_update_grid_search = [1, 5, 10]

step_size_grid_search = [1]
grad_update_grid_search = [1]

clusters_grid = [3] 

base_seed = 301

def set_seeds(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
Implmenting simple models for classification with a latent category.
Some of the code is adapted from Goncalo Correia's code: https://github.com/deep-spin/simple-discrete-vae

TODO:
- Multiple gradient update steps on the backward pass (STE and SPIGOT)
"""


def train_model(model, X, y, z, X_eval, y_eval, z_eval, X_test, y_test, z_test, \
                num_steps, save_every_steps, print_every_steps, \
                exact_per_instance=False):

    print('N parameters', count_parameters(model))
    print(model)

    # get name of model to look up lr
    if learning_rate == 'best':
        model_name = type(model).__name__
        if model_name == 'LatentClassifier':
            model_name = type(model.latent_model).__name__
        learning_rate = BEST_LRS[model_name]

    # mm = torch.nn.Bilinear(3, 100, 2)
    # print('xZ', X.shape, z.shape)
    # y_hat = mm(z, X) #.requires_grad_()
    # loss = loss_function(y_hat, y)
    # loss.backward(create_graph=True)

    if device == 'cuda':
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    train_accuracies = []
    eval_accuracies = []
    test_accuracies = []
    train_v_measures = []
    eval_v_measures = []
    test_v_measures = []

    reward_stats = RunningStats()

    for step in range(num_steps+1):
        if isinstance(model, ExactSearchClassifier):
            if exact_per_instance:
                # We will train the model for each instance separately
                z_hat = torch.zeros(z.shape, device=device)
                loss = 0
                losses_temp = 0
                for i, (X_temp, y_temp, z_temp) in enumerate(zip(X, y, z)):
                    loss_temp, z_hat_temp = model(X_temp.unsqueeze(0), y_temp.unsqueeze(0), loss_function)
                    loss += loss_temp
                    z_hat[i] = z_hat_temp
                loss /= len(X)
            else:
                # Exact search for the whole batch:
                loss, z_hat = model(X, y, loss_function)

        elif isinstance(model, REINFORCEClassifier):
            loss, z_hat = model(X, y, loss_function, reward_stats)

        elif isinstance(model, LatentClassifier):
            y_hat, z_hat = model(X, z, y)
            loss = loss_function(y_hat, y)
        else:
            y_hat, z_hat = model(X, z)
            loss = loss_function(y_hat, y)

        loss.backward()

        optimizer.step()
        model.zero_grad()

        if step % save_every_steps == 0:
            if isinstance(model, ExactSearchClassifier) \
                or isinstance(model, REINFORCEClassifier) \
                or isinstance(model, LatentClassifier):
                model.set_eval()
            eval_accuracy, eval_v_measure = eval_model(model, X_eval, y_eval, z_eval)
            test_accuracy, test_v_measure = eval_model(model, X_test, y_test, z_test)
            if isinstance(model, ExactSearchClassifier) \
                or isinstance(model, REINFORCEClassifier) \
                or isinstance(model, LatentClassifier):
                model.set_train()
            eval_accuracy = eval_accuracy / len(y_eval)
            test_accuracy = test_accuracy / len(y_test)

            losses.append(loss.detach().item())
            eval_accuracies.append(eval_accuracy)
            test_accuracies.append(test_accuracy)
            eval_v_measures.append(eval_v_measure)
            test_v_measures.append(test_v_measure)

            train_accuracy = 0
            if not isinstance(model, REINFORCEClassifier):
                train_accuracy = len((torch.argmax(y_hat.detach(), dim=-1) == y.detach()).nonzero()) / len(y)
            train_accuracies.append(train_accuracy)

            train_v_measure = 0
            if z_hat is not None:
                train_v_measure = metrics.v_measure_score(z.detach().to('cpu'), z_hat.detach().to('cpu'))
                train_v_measures.append(train_v_measure)

            if step % print_every_steps == 0:
                print('--------------------------')
                print(time.strftime("%x %X", time.gmtime()))
                print('Step {}: loss {} '.format(step, loss))
                print('train accuracy {}, train v-measure {}'.format(train_accuracy, train_v_measure))
                print('eval accuracy {}, eval v-measure {}'.format(eval_accuracy, eval_v_measure))
                print('test accuracy {}, test v-measure {}'.format(test_accuracy, test_v_measure))

            if save_model:
                step_size = 0
                gradient_update_steps = 0
                model_name = type(model).__name__
                if isinstance(model, LatentClassifier):
                    if hasattr(model.latent_model, 'latent_config'):
                        step_size = model.latent_model.latent_config['step_size']
                        gradient_update_steps = model.latent_model.latent_config['gradient_update_steps']
                    latent_name = type(model.latent_model).__name__
                    model_name += '-' + latent_name
                model_file = f"results/model_{model_name}_{learning_rate}_{step_size}_step_{gradient_update_steps}_grad_epoch_{step}.pt"
                torch.save(model, model_file)
                print('saved model:', model_file)

    return losses, (train_accuracies, eval_accuracies, test_accuracies), (train_v_measures, eval_v_measures, test_v_measures)


def eval_model(model, X_eval, y_eval, z_eval):
    with torch.no_grad():
        y_hat, z_hat = model(X_eval, z_eval)
        y_hat = torch.argmax(y_hat, dim=-1)
        accuracy = len((y_hat == y_eval).nonzero())
        v_measure = 0
        if z_hat is not None:
            v_measure = metrics.v_measure_score(z_eval.detach().to('cpu'), z_hat.detach().to('cpu'))
        return accuracy, v_measure


def latent_model_configurations():
    return [
            # (LatentSTEZero, 'STE-Zero', '-', True, True),
            # (LatentSPIGOTZero, 'SPIGOT-Zero', '-', True, True),
            # (LatentSPIGOTCE, 'SPIGOT-CrossEntr-ArgmaxTest', '-', True, True),
            # (LatentSPIGOTEG, 'SPIGOT-ExpGrad-ArgmaxTest', '-', True, True),
            # (LatentSPIGOTCE, 'SPIGOT-CrossEntr-RelaxedTest', '-', False, True),
            # (LatentSPIGOTEG, 'SPIGOT-ExpGrad-RelaxedTest', '-', False, True),
            (LatentSPIGOTCEArgmax, 'SPIGOT-CrossEntr-ArgmaxFwd', '-', True, True), # need full grid search
            (LatentSPIGOTEGArgmax, 'SPIGOT-ExpGrad-ArgmaxFwd', '-', True, True), # need full grid search

            (LatentSoftmax, 'Softmax-ArgmaxTest', '--', True, False),
            (LatentSparsemax, 'Sparsemax-ArgmaxTest', '--', True, False),
            # (LatentSoftmax, 'Softmax-RelaxedTest', '--', False, False), # LR grid search
            # (LatentSparsemax, 'Sparsemax-RelaxedTest', '--', False, False), # LR grid search

            (LatentSTEIdentity, 'STE-I', '-', True, True), # need full grid search
            (LatentSTESoftmax, 'STE-S', '-', True, False), # LR grid search
            (LatentSPIGOT, 'SPIGOT', '-', True, True), # need full grid search

            # (LatentGumbelSoftmax, 'Gumbel Softmax-ArgmaxTest', '-.', True, False),
            # (LatentGumbelSoftmax, 'Gumbel Softmax-RelaxedTest', '-.', False, False), # LR grid search
            (LatentGumbelSTE, 'Gumbel STE', '-.', True, False), # LR grid search
           ]


def build_latent_model(latent_model, n_features, n_classes, n_clusters, \
                        argmax_test_time):
    encoder = Encoder(n_features, n_clusters)
    decoder = Decoder(n_features, n_clusters, n_classes)
    model = LatentClassifier(encoder, decoder, latent_model, argmax_test_time)
    return model


def plot_results(results_map, filename='results.png'):
    """
    Input:
    Map with
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    for (lbl, result) in results_map.items():
        print('key:', lbl)
        print('result:', result)
        ax[0].plot(result['eval_accuracies'][0], label=lbl, linestyle=result['linestyle'])
        ax[1].plot(result['eval_vmeasures'][0], label=lbl, linestyle=result['linestyle'])

    ax[0].set(xlabel='Steps', ylabel='Accuracy')
    ax[0].grid()
    ax[0].legend()

    ax[1].set(xlabel='Steps', ylabel='V-Measure')
    ax[1].grid()
    ax[1].legend()

    fig.savefig(filename)


def plot_results_range(results_map, show_plots=None, filename='results.png'):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    for (lbl, result) in results_map.items():
        if show_plots is None or lbl in show_plots:
            print('---------------------')
            print(lbl)
            print('Accuracies:')
            plot_range(result['eval_accuracies'], ax[0], lbl, result['linestyle'], True)
            print('V-measures:')
            plot_range(result['eval_vmeasures'], ax[1], lbl, result['linestyle'])
            print('---------------------')

    ax[0].set(xlabel='Steps', ylabel='Accuracy')
    ax[0].grid()
    ax[0].legend()

    ax[1].set(xlabel='Steps', ylabel='V-Measure')
    ax[1].grid()
    ax[1].legend()

    fig.savefig(filename)

    plt.show()


def plot_range(curve, ax, label, linestyle, print_info=False):
    median_curve = np.percentile(curve, 50, axis=0)
    lower_curve = np.percentile(curve, 25, axis=0)
    upper_curve = np.percentile(curve, 75, axis=0)
    mean_curve = np.mean(curve, axis=0)
    ax.plot(median_curve, label=label, linestyle=linestyle)
    if print_info:
        print('mean:', mean_curve)
        print('max mean:', np.max(mean_curve))
    ax.fill_between(range(len(median_curve)), lower_curve, upper_curve, alpha=0.25)


def append_results(results, losses, accuracies, vmeasures, label, line_style):
    if not label in results.keys():
        results[label] = {'linestyle': line_style, 'losses': [], \
                            'train_accuracies': [], 'eval_accuracies': [], 'test_accuracies': [], \
                            'train_vmeasures': [], 'eval_vmeasures': [], 'test_vmeasures': []}
    results[label]['losses'].append(losses)
    (train_accuracies, eval_accuracies, test_accuracies) = accuracies
    (train_vmeasures, eval_vmeasures, test_vmeasures) = vmeasures
    results[label]['train_accuracies'].append(train_accuracies)
    results[label]['eval_accuracies'].append(eval_accuracies)
    results[label]['test_accuracies'].append(test_accuracies)
    results[label]['train_vmeasures'].append(train_vmeasures)
    results[label]['eval_vmeasures'].append(eval_vmeasures)
    results[label]['test_vmeasures'].append(test_vmeasures)



def main():
    # for n_clusters in [3, 5, 10, 20]:
    for n_clusters in clusters_grid:
        for n_features in [100]:
            for step_size in step_size_grid_search:
                for gradient_update_steps in grad_update_grid_search:
                    run_experiments(n_clusters, n_features, step_size, gradient_update_steps)


def run_experiments(n_clusters, n_features, step_size, gradient_update_steps):
    """
    Set the config
    """

    # This data always has 2 classes, this is set here to avoid hardcoding in the models
    n_classes = 2


    if DEMO:
        # Data settings
        n_samples = 60
        train_size = 50
        test_size = n_samples - train_size
        # Training settings
        num_steps = 10
        print_every = 10
        save_every = 5
    else:
        # Data settings
        n_samples = 6000
        train_size = 5000
        test_size = n_samples - train_size
        # Training settings
        num_steps = 10000
        print_every = 1000
        save_every = 500
    # Other setup
    exact_per_instance = True
    # Models with latent variables
    latent_model_config = latent_model_configurations()

    """
    Prepare the data
    """
    set_seeds(42)
    X, y, z, clusters, W, b = make_latent_triples(n_samples, n_features, n_clusters)
    # Turn the -1s to 0s in the target variable
    y = (y+1)/2
    y = y.type(torch.LongTensor)

    # Split the train and evaluation data
    X_train, X_eval = X[0:train_size], X[train_size:]
    y_train, y_eval = y[0:train_size], y[train_size:]
    z_train, z_eval = z[0:train_size], z[train_size:]

    X_test, y_test, z_test, _, _, _ = make_latent_triples(test_size, n_features, n_clusters, clusters, W, b)
    # Turn the -1s to 0s in the target variable
    y_test = (y_test+1)/2
    y_test = y_test.type(torch.LongTensor)

    if device == 'cuda':
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        z_train = z_train.cuda()
        X_eval = X_eval.cuda()
        y_eval = y_eval.cuda()
        z_eval = z_eval.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        z_test = z_test.cuda()

    """
    Train all available models, save losses and accuracies for every 'save_every' steps
    """
    results = {}

    count = 0

    for i in range(n_runs):
        count += 1
        print('============================= RUNNING', i, count)

        if run_baselines:
            # Linear model without latent variables
            # set_seeds(base_seed + i)
            # model = LinearClassifier(n_features, n_classes)
            # losses, accuracies, vmeasures = \
            #     train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
            #         X_test, y_test, z_test, \
            #         num_steps, save_every, print_every)
            # append_results(results, losses, accuracies, vmeasures, 'Linear model', '--')

            # Model with gold labels appended
            print('running baseline')
            set_seeds(base_seed + i)
            model = GoldLatentClassifier(n_features, n_clusters, n_classes)
            losses, accuracies, vmeasures = \
                train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
                    X_test, y_test, z_test, \
                    num_steps, save_every, print_every)
            append_results(results, losses, accuracies, vmeasures, 'Gold labels', '--')


        if run_reinforce:
            # Model with REINFORCE - no baseline
            set_seeds(base_seed + i)
            encoder = Encoder(n_features, n_clusters)
            decoder = Decoder(n_features, n_clusters, n_classes)
            model = REINFORCEClassifier(encoder, decoder, False)
            losses, accuracies, vmeasures = \
                train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
                    X_test, y_test, z_test, \
                    num_steps, save_every, print_every)
            append_results(results, losses, accuracies, vmeasures, 'REINFORCE No Baseline', '-.')


            # Model with REINFORCE - with running average baseline
            set_seeds(base_seed + i)
            encoder = Encoder(n_features, n_clusters)
            decoder = Decoder(n_features, n_clusters, n_classes)
            model = REINFORCEClassifier(encoder, decoder, True)
            losses, accuracies, vmeasures = \
                train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
                    X_test, y_test, z_test, \
                    num_steps, save_every, print_every)
            append_results(results, losses, accuracies, vmeasures, 'REINFORCE With Baseline', '-.')


        # Models with latent variables
        if run_latent_model:
            for (latent_model_class, label, linestyle, argmax_test_time, latent_update_with_params) in latent_model_config:
                latent_config = {'step_size': step_size, 'gradient_update_steps': gradient_update_steps}
                print('runnig latent...')
                print(latent_model_class)
                set_seeds(base_seed + i)
                latent_model = latent_model_class()
                if latent_update_with_params:
                    latent_model.set_latent_config(latent_config)
                model = build_latent_model(latent_model, n_features, n_classes, n_clusters, \
                                            argmax_test_time)
                losses, accuracies, vmeasures = \
                    train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
                        X_test, y_test, z_test, \
                        num_steps, save_every, print_every)
                label = f"{label}-st{step_size}-gr{gradient_update_steps}"
                append_results(results, losses, accuracies, vmeasures, label, linestyle)



        if run_minimum_risk_training:
            # Model with weighted sum of losses for all possible values of z
            encoder = Encoder(n_features, n_clusters)
            decoder = Decoder(n_features, n_clusters, n_classes)
            model = ExactSearchClassifier(encoder, decoder, 'sum')
            losses, accuracies, vmeasures = \
                train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
            X_test, y_test, z_test, \
            num_steps, save_every, print_every, exact_per_instance)
            append_results(results, losses, accuracies, vmeasures, 'MRT', '--')


        if run_exact:
            # # Model with exact search for the best z for the downstream task
            encoder = Encoder(n_features, n_clusters)
            decoder = Decoder(n_features, n_clusters, n_classes)
            model = ExactSearchClassifier(encoder, decoder, 'best')
            losses, accuracies, vmeasures = \
                train_model(model, X_train, y_train, z_train, X_eval, y_eval, z_eval, \
            X_test, y_test, z_test, \
            num_steps, save_every, print_every, exact_per_instance)
            append_results(results, losses, accuracies, vmeasures, 'Best z exact', '--')


    print(results)

    """
    Plot the results
    """
    # plot_results(results)
    # plot_results_range(results)

    # Save the results to file:
    filename = \
    f"results/gridsearch_results_{n_clusters}_clusters_{n_features}_features_{learning_rate}_{step_size}_step_{gradient_update_steps}_grad_multirun.txt"
    with open(filename, 'w') as outfile:
        json.dump(results, outfile)
    print(filename)






if __name__ == '__main__':
    main()

