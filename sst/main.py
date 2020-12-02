import os
import time
import random
# random.seed(777)

import torch
# torch.manual_seed(777)
import matplotlib.pyplot as plt

from torchtext import data
from torchtext.datasets import SST

from models import LSTMBaseline, LatentSyntaxClassifier, use_contextual_embeddings, use_attention


device = 'cuda' if torch.cuda.is_available() else 'cpu'

default_learning_rate = 0.0001
default_run_baseline = True
argmax_test_time = True
save_model_per_epoch = False
use_nonprojective_parsing = False

anneal_lr = False

# Set this to True for runnign with small parameters, in order to verify whether the cod eis working
DEMO = False
num_epochs = 50

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_results(losses, train_acc, dev_acc, test_acc):
    plt.plot(losses, label='losses')
    plt.plot(train_acc, label='train_acc')
    plt.plot(dev_acc, label='dev_acc')
    plt.plot(test_acc, label='test_acc')

    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.legend()
    plt.show()


def write_line_to_file(filename, *text):
    text = ' '.join(map(str, text))
    print(text)
    with open(filename, 'a', encoding="utf-8") as out:
        out.write(text)
        out.write('\n')

def printable_time():
    return time.strftime("%x %X", time.gmtime())

class BinarizedSST(SST):
    def __init__(self, path, text_field, label_field, fine_grained=None, subtrees=None, **kwargs):

        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
            return {'0': 'negative', '1': 'negative',
                    '2': 'neutral',
                    '3': 'positive', '4': 'positive', None: None}[label]

        label_field.preprocessing = data.Pipeline(get_label_str)

        with open(os.path.expanduser(path)) as f:
            examples = (data.Example.fromtree(line, fields) for line in f)
            examples = [x for x in examples if x.label != 'neutral']

        print("N samples:", len(examples))
        print(examples[0].text, examples[0].label)

        data.Dataset.__init__(self, examples, fields, **kwargs)


def main(run_baseline, learning_rate, hidden_size, hidden_size_out, \
            latent_step_size, latent_grad_updates, latent_model_type=None):

    if DEMO:
        sst_train, sst_dev, sst_test = BinarizedSST.iters(device=device, vectors='glove.6B.50d')
        num_epochs = 3
    else:
        sst_train, sst_dev, sst_test = BinarizedSST.iters(device=device, vectors='glove.840B.300d', batch_size=8)

    # initialize embeddings with this
    vectors = sst_train.dataset.fields['text'].vocab.vectors
    n_classes = 2
    embedding_size = vectors.shape[1]
    # hidden_size = embedding_size
    # hidden_size = 50

    # ====== < BASELINES ======
    bow_baseline = torch.nn.Sequential(
        torch.nn.EmbeddingBag.from_pretrained(vectors),
        torch.nn.Linear(embedding_size, hidden_size_out),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size_out, n_classes)
    )

    lstm_baseline = LSTMBaseline(vectors, embedding_size, hidden_size, hidden_size_out, n_classes)
    # ====== BASELINES > ======

    # ====== Latent Models ======
    all_latent_model_types =   [
                            'ste_identity', # has update steps
                            'ste_marginals',
                            'spigot', # has update steps
                            # 'spigot_ce', # relaxed on forward; NOT USED ANYMORE - replaced with spigot_ce_argmax
                            # 'spigot_eg', # relaxed on forward; NOT USED ANYMORE - replaced with spigot_ce_argmax
                            'spigot_ce_argmax', # argmax on forward # has update steps
                            'spigot_eg_argmax', # argmax on forward # has update steps
                            'marginals', # relaxed on forward
                            'sparsemap', # relaxed on forward
                            'perturb_and_map',
                            'argmax',
                            ]

    baseline_models = [lstm_baseline, bow_baseline]

    if run_baseline:
        items = baseline_models
    else:
        if latent_model_type:
            items = [latent_model_type]
        else:
            items = latent_model_types

    for item in items:
        emb_type = 'emblstm' if use_contextual_embeddings else 'emb'
        using_attn = 'attn' if use_attention else ''
        if run_baseline:
            model_name = type(item).__name__
            model = item    
            results_file = f"results/results_2_{'np' if use_nonprojective_parsing else 'p'}_baseline_{model_name}_{embedding_size}_{hidden_size}_{hidden_size_out}_{learning_rate}_{emb_type}{using_attn}_adamw.txt"
            model_file = f"results/model_2_{'np' if use_nonprojective_parsing else 'p'}_{model_name}_{embedding_size}_{hidden_size}_{hidden_size_out}_{learning_rate}_{emb_type}{using_attn}_adamw_epoch_EPOCH.pt"
        else:
            latent_model_type = item
            results_file = f"results/results_2_{'np' if use_nonprojective_parsing else 'p'}_{latent_model_type}_{embedding_size}_{hidden_size}_{hidden_size_out}_{learning_rate}_{latent_step_size}_{latent_grad_updates}_{argmax_test_time}_{emb_type}{using_attn}_adamw.txt"
            model_file = f"results/model_2_{'np' if use_nonprojective_parsing else 'p'}_{latent_model_type}_{embedding_size}_{hidden_size}_{hidden_size_out}_{learning_rate}_{latent_step_size}_{latent_grad_updates}_{argmax_test_time}_{emb_type}{using_attn}_adamw_epoch_EPOCH.pt"

            latent_syntax_model = LatentSyntaxClassifier(vectors, embedding_size, hidden_size, hidden_size_out, n_classes, latent_model_type)

            model = latent_syntax_model
            model_name = latent_model_type

        run_model(sst_train, sst_dev, sst_test, model, model_name, results_file, model_file, 
                run_baseline, learning_rate, latent_step_size, latent_grad_updates)

def run_model(sst_train, sst_dev, sst_test, \
                model, model_name, results_file, model_file, run_baseline, learning_rate, \
                latent_step_size, latent_grad_updates):                
    latent_config = {'step_size': latent_step_size, 'gradient_update_steps': latent_grad_updates}

    current_learning_rate = learning_rate
    previous_accuracy = 0

    write_line_to_file(results_file, '===================')
    write_line_to_file(results_file, printable_time(), 'Started training of model:', model_name)
    write_line_to_file(results_file, 'N parameters', count_parameters(model))

    if device == 'cuda':
        model.cuda()

    write_line_to_file(results_file, 'Model:', model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    losses = []
    accuracies_dev = []
    accuracies_test = []
    accuracies_train = []

    for epoch in range(num_epochs):
        write_line_to_file(results_file, printable_time(), ': Starting epoch', epoch)
        c = 0

        n_correct_train = 0
        total_train = 0
        for batch in sst_train:
            if DEMO:
                c+=1
                if c > 5:
                    break
            model.train()
            optimizer.zero_grad()
            x = batch.text
            y = batch.label - 1  # whyy is it 1,2 and not 0,1??
            y = torch.nn.functional.one_hot(y).float()

            if run_baseline:
                pred = model(x.t())
            else:
                pred = model(x.t(), y, latent_config)
            # print(x, y)
            if y.shape[1] == 2: # necessary because of a problem with some instances: TODO: FIX THIS!!!
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
                loss.backward()
                # print('LOSS:', loss)
                optimizer.step()

            pred_ind = pred.argmax(dim=1)
            # print('predd', pred_ind.shape, y.shape)
            n_correct_train += (pred_ind == y.argmax(dim=1)).sum().item()
            total_train += len(pred_ind)
        accuracy_train = n_correct_train / total_train

        n_correct = 0
        total = 0
        for batch in sst_dev:
            model.eval()
            x = batch.text
            y = batch.label - 1  # whyy is it 1,2 and not 0,1??
            if run_baseline:
                pred = model(x.t()).argmax(dim=1)
            else:
                pred = model(x.t(), y, latent_config, argmax_test_time).argmax(dim=1)
            n_correct += (pred == y).sum().item()
            total += len(pred)
        accuracy = n_correct / total

        n_correct_test = 0
        total_test = 0
        for batch in sst_test:
            model.eval()
            x = batch.text
            y = batch.label - 1  # whyy is it 1,2 and not 0,1??
            if run_baseline:
                pred = model(x.t()).argmax(dim=1)
            else:
                pred = model(x.t(), y, latent_config, argmax_test_time).argmax(dim=1)
            n_correct_test += (pred == y).sum().item()
            total_test += len(pred)
        accuracy_test = n_correct_test / total_test

        if anneal_lr:
            if previous_accuracy > accuracy:
                current_learning_rate = 0.9*current_learning_rate
                write_line_to_file(results_file, 'Updating learning_rate:', current_learning_rate, ':', previous_accuracy, accuracy )
                for g in optimizer.param_groups:
                    g['lr'] = current_learning_rate
            previous_accuracy = accuracy


        losses.append(loss.item())
        accuracies_dev.append(accuracy)
        accuracies_test.append(accuracy_test)
        accuracies_train.append(accuracy_train)

        write_line_to_file(results_file, 'Loss:', loss.item())
        write_line_to_file(results_file, 'Train Accuracy:', accuracy_train)
        write_line_to_file(results_file, 'Validation Accuracy:', accuracy)
        write_line_to_file(results_file, 'Test Accuracy:', accuracy_test)
        write_line_to_file(results_file, '===============================')

        if save_model_per_epoch:
            torch.save(model, model_file.replace('EPOCH', str(epoch)))




    torch.save(model, model_file.replace('EPOCH', 'LAST'))

    write_line_to_file(results_file, 'Model:', model)
    write_line_to_file(results_file, 'Losses:', losses)
    write_line_to_file(results_file, 'Accuracies Train:', accuracies_train) 
    write_line_to_file(results_file, 'Accuracies Dev:', accuracies_dev)
    write_line_to_file(results_file, 'Accuracies Test:', accuracies_test)
    write_line_to_file(results_file, 'Latent:', model_name)

    plot_results(losses, accuracies_train, accuracies_dev, accuracies_test)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parameters for running the models.')
    parser.add_argument('--baseline', type=bool, help='Whether to run baseline', default=False)
    parser.add_argument('--learning_rate', type=float, help='The optimizer learning rate.')
    parser.add_argument('--hidden_size', type=int, help='The model hidden size.')
    parser.add_argument('--hidden_size_out', type=int, help='Decoder hidden size.')
    parser.add_argument('--latent_step_size', type=float, 
        help='Step size for updating the latent variable from the downstream loss.')
    parser.add_argument('--latent_grad_updates', type=int, 
        help='Number of gradient update steps for updating the latent variable from the downstream loss.')
    parser.add_argument('--latent_model_type', type=str, 
        help='Run for a specific latent model. Otherwise it will execute for all latent models.')

    args = parser.parse_args()
    print('args:', args)

    main(args.baseline, args.learning_rate, args.hidden_size, args.hidden_size_out, \
            args.latent_step_size, args.latent_grad_updates, args.latent_model_type)





