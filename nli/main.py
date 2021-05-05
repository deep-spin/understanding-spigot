import numpy as np
import torch
from torchtext.datasets import SNLI

from sorted_snli import SortedSNLI
from filtered_snli import FilteredSNLI, max_sent_length
from models import Decomp
from helpers import printable_time, write_line_to_file



toy = False
epochs = 30
additional_epochs = 30
save_model_each_epoch = False
demo = False
filtered_data = True
default_batch_size = 64

run_id = '_1'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main(baseline, learning_rate, hidden_size, dropout, 
         latent_model_type, latent_step_size, latent_grad_updates, read_from_epoch=-1):
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_latent_trees = not baseline
    latent_config = {'step_size': latent_step_size, 'gradient_update_steps': latent_grad_updates}

    if baseline:
        model_name = 'Baseline'
    else: 
        model_name = latent_model_type

    results_file = f"results/results{run_id}_{model_name}_{hidden_size}_{learning_rate}_{dropout}_{max_sent_length if filtered_data else ''}_{latent_step_size}_{latent_grad_updates}_adamw.txt"
    model_file = f"results/model{run_id}_{model_name}_{hidden_size}_{learning_rate}_{dropout}_{max_sent_length if filtered_data else ''}_{latent_step_size}_{latent_grad_updates}_adamw_epoch_EPOCH.pt"

    if toy:
        vectors_fn = 'glove.6B.50d'
        Loader = SortedSNLI
        batch_size = 3
    else:
        vectors_fn = 'glove.840B.300d'
        Loader = FilteredSNLI if filtered_data else SNLI
        batch_size = 10 if demo else default_batch_size
        # batch_size = 10

    write_line_to_file(results_file, 'Reading data...', Loader)
    train, valid, test = Loader.iters(vectors=vectors_fn,
                                      batch_size=batch_size,
                                      trees=False,
                                      device=device)

    write_line_to_file(results_file, f'max_sent_len={(max_sent_length if filtered_data else -1)} Dataset size: TRAIN={len(train)*batch_size} VALID={len(valid)*batch_size} TEST={len(test)*batch_size}')

    vocab = train.dataset.fields['premise'].vocab
    vectors = vocab.vectors


    if read_from_epoch > 0:
        nn = torch.load(model_file.replace('EPOCH', str(read_from_epoch)))
    else:
        nn = Decomp(vectors, n_hid=hidden_size, p_drop=dropout, latent_type=latent_model_type, latent_config=latent_config)

    if device == 'cuda':
        nn.cuda()

    write_line_to_file(results_file, "N parameters", count_parameters(nn))

    opt = torch.optim.AdamW(nn.parameters(), lr=learning_rate)

    losses = []

    all_losses = []
    accuracies_dev = []
    accuracies_train = []
    accuracies_test = []
    range_from = 0
    range_to = epochs
    if read_from_epoch:
        range_from = read_from_epoch+1
        range_to = read_from_epoch+1+additional_epochs
    for epoch in range(range_from, range_to):
        write_line_to_file(results_file, printable_time(), ': Starting epoch', epoch+1)
        acc_train = 0
        total_train = 0
        if demo:
            count = 0
        for k, batch in enumerate(train):
            if demo:
                count +=1 
                if count > 5:
                    break
            y_true = batch.label - 1

            nn.train()

            s_pred = nn(batch)
            loss = torch.nn.functional.cross_entropy(s_pred, y_true)

            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

            acc_train += torch.sum(y_true == s_pred.argmax(dim=1)).item()
            total_train += y_true.shape[0]

            # if k % 1000 == 0:
            #     print(np.mean(losses))
            #     losses = []

        acc_dev = 0
        total_dev = 0
        for k, batch in enumerate(valid):
            nn.eval()
            s_pred = nn(batch)
            y_true = batch.label - 1
            _, y_pred = s_pred.max(dim=1)
            acc_dev += torch.sum(y_true == y_pred).item()
            total_dev += y_true.shape[0]

        acc_test = 0
        total_test = 0
        for k, batch in enumerate(test):
            nn.eval()
            s_pred = nn(batch)
            y_true = batch.label - 1
            _, y_pred = s_pred.max(dim=1)
            acc_test += torch.sum(y_true == y_pred).item()
            total_test += y_true.shape[0]

        acc_train = acc_train / total_train
        acc_dev = acc_dev / total_dev
        acc_test = acc_test / total_test

        all_losses.append(np.mean(losses))
        accuracies_dev.append(acc_dev)
        accuracies_test.append(acc_test)
        accuracies_train.append(acc_train)
        write_line_to_file(results_file, "Loss: ", np.mean(losses))
        write_line_to_file(results_file, "Train Accuracy: ", acc_train)
        write_line_to_file(results_file, "Validation Accuracy: ", acc_dev)
        write_line_to_file(results_file, "Test Accuracy: ", acc_test)
        write_line_to_file(results_file, '===============================')

        if save_model_each_epoch:
            torch.save(nn, model_file.replace('EPOCH', str(epoch)))

    if not save_model_each_epoch:
        torch.save(nn, model_file.replace('EPOCH', str(epoch)))
    write_line_to_file(results_file, 'Model:', nn)
    write_line_to_file(results_file, 'Losses:', all_losses)
    write_line_to_file(results_file, 'Accuracies Train:', accuracies_train) 
    write_line_to_file(results_file, 'Accuracies Dev:', accuracies_dev)
    write_line_to_file(results_file, 'Accuracies Test:', accuracies_test)
    write_line_to_file(results_file, 'Latent:', model_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parameters for running the models.')
    parser.add_argument('--baseline', type=bool, help='Whether to run baseline', default=False)
    parser.add_argument('--learning_rate', type=float, help='The optimizer learning rate.')
    parser.add_argument('--hidden_size', type=int, help='The model hidden size.')
    parser.add_argument('--dropout', type=float, help='The model dropout.')
    parser.add_argument('--latent_model_type', type=str, 
        help='Run for a specific latent model. Otherwise it will execute for all latent models.')
    parser.add_argument('--latent_step_size', type=float, 
        help='Step size for updating the latent variable from the downstream loss.')
    parser.add_argument('--latent_grad_updates', type=int, 
        help='Number of gradient update steps for updating the latent variable from the downstream loss.')
    parser.add_argument('--run_from_epoch', type=int, default=-1,
        help='If we want the model to continue from a specific epoch, pass it as a parameter.')

    args = parser.parse_args()
    print('args:', args)

    main(args.baseline, args.learning_rate, args.hidden_size, args.dropout,  
        args.latent_model_type, args.latent_step_size, args.latent_grad_updates, args.run_from_epoch)

