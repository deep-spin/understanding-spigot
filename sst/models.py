import copy

import torch
from torch.distributions import Uniform

from structured_ste import (ste_identity, ste_marginals, spigot, sparsemap,
                            spigot_ce, spigot_eg, spigot_ce_argmax,
                            spigot_eg_argmax)

from torch_struct import DependencyCRF

# from sparsemap import sparsemap_batched

# If True, use output from LSTM for the decoder
# If False, use word embeddings for the decoder
use_contextual_embeddings = True

use_attention = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_argmax(scores, X=None):
    scores = scores.detach()
    dist = DependencyCRF(scores)
    trees = dist.argmax.detach()
    return trees


class LSTMBaseline(torch.nn.Module):
    def __init__(self, pretrained_embeddings, embedding_size, hidden_size, hidden_size_out, n_classes):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(pretrained_embeddings)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size_out),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size_out, n_classes)
        )

    def forward(self, inputs):
        # print('inputs:', inputs.shape)
        embedded = self.embeddings(inputs)
        # print('embedded:', embedded.shape)
        lstm_out, _ = self.lstm(embedded)
        summed = torch.sum(lstm_out, dim=1)
        # print('lstm_out:', lstm_out.shape)
        output = self.mlp(summed)
        # print('output:', output.shaspe)
        return output


class DependencyParser(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            )

    def calc_arc_scores(self, X):
        n = X.shape[1]
        X1 = X.unsqueeze(1).repeat(1, X.shape[1], 1, 1)
        X2 = X.unsqueeze(2).repeat(1, 1, X.shape[1], 1)
        # Build a matrix with each of the concatenated scores
        arc_vectors = torch.cat((X1.transpose(1,2), X2.transpose(1,2)), dim=3)

        # Calculate the arc scores
        arc_scores = self.mlp(arc_vectors).squeeze(-1)
        # print('arc_scores:', arc_scores.shape)

        return arc_scores


    def forward(self, encoded_sequence, latent_type, latent_config, decoder, X, y):
        """
        The possbile values for latent_type are:
        - argmax - MAP on forward
        - marginals - MArginals on forward
        - ste_identity - MAP on forward, Identity on backward
        - ste_marginals - MAP on forward, marginals on backward
        - perturb_and_map - Gumbel noise + MAP on forward, marginals on backward
        """

        # # Calculate the arc scores
        arc_scores = self.calc_arc_scores(encoded_sequence)

        X_ = encoded_sequence if use_contextual_embeddings else X
        if latent_type == 'marginals':
            dist = DependencyCRF(arc_scores)
            parser_output = dist.marginals
        elif latent_type == 'ste_identity':
            parser_output = ste_identity(arc_scores, decoder, X_, y, latent_config)
        elif latent_type == 'ste_marginals':
            parser_output = ste_marginals(arc_scores)
        elif latent_type == 'sparsemap':
            parser_output = sparsemap(arc_scores)
        elif latent_type == 'spigot':
            parser_output = spigot(arc_scores, decoder, X_, y, latent_config)
        elif latent_type == 'spigot_ce':
            parser_output = spigot_ce(arc_scores, decoder, X_, y, latent_config)
        elif latent_type == 'spigot_eg':
            parser_output = spigot_eg(arc_scores, decoder, X_, y, latent_config)
        elif latent_type == 'spigot_ce_argmax':
            parser_output = spigot_ce_argmax(arc_scores, decoder, X_, y, latent_config)
        elif latent_type == 'spigot_eg_argmax':
            parser_output = spigot_eg_argmax(arc_scores, decoder, X_, y, latent_config)
        elif latent_type == 'perturb_and_map':
            # Perturn-and-MAP equals:
            # Add Gumbel noise to the arc scores and apply STE with marginals
            m = Uniform(0, 1)
            u = m.sample(arc_scores.shape)
            g = -torch.log(-torch.log(u))
            if device == 'cuda':
                g = g.cuda()
            parser_output = ste_marginals(arc_scores+g)
        else: # assuming argmax as default

            parser_output = get_argmax(arc_scores, X)

        return parser_output


class LatentSyntaxClassifier(torch.nn.Module):
    def __init__(self, pretrained_embeddings, embedding_size, hidden_size, hidden_size_out, n_classes, latent_type):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(pretrained_embeddings)
        self.encoder = torch.nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)

        decoder_input_size = hidden_size*2 if use_contextual_embeddings else embedding_size*2
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(decoder_input_size, hidden_size_out),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size_out, n_classes)
        )
        self.parser = DependencyParser(hidden_size)
        self.latent_type = latent_type
        self.attn_query = torch.nn.Linear(hidden_size*2 if use_contextual_embeddings else embedding_size*2, 1)

    def tree_representation(self, parser_output, parser_base_vectors):
        avg_heads = torch.einsum('bnm,bnh->bnh', parser_output, parser_base_vectors)
        tree_representation = torch.cat((parser_base_vectors, avg_heads), dim=2)
        
        if use_attention:
            # print('using attn:')
            # print('tree_representation:', tree_representation.shape)
            attn_scores = self.attn_query(tree_representation)
            # print('attn_scores:', attn_scores.shape)
            attn_proba = torch.softmax(attn_scores, dim=-1)
            # print('attn_proba:', attn_proba.shape)
            tree_representation = torch.einsum('bnm,bnh->bm', tree_representation, attn_proba)
            # print('tree_representation:', tree_representation.shape)
        else:
            tree_representation = torch.sum(tree_representation, dim=1) 
        return tree_representation

    def forward(self, inputs, y, latent_config, latent_argmax=False, show_tree=False):
        embedded = self.embeddings(inputs)
        encoded, _ = self.encoder(embedded)
        latent_type = self.latent_type
        if latent_argmax:
            latent_type = 'argmax' # This is used for passing argmax on test time
        parser_output = self.parser(encoded, latent_type, latent_config, self.decoder, embedded, y)
        # Combine the parser result with the arc vectors to obtain a representation of the tree
        tree_representation = self.tree_representation(parser_output, (encoded if use_contextual_embeddings else embedded))
        output = self.decoder(tree_representation)
        if show_tree:
            return output, parser_output
        else:
            return output



