import numpy as np
import torch

from torch.distributions import Uniform
from torch_struct import NonProjectiveDependencyCRF as DependencyCRF

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sparsemap import argmax_batched
from structured_ste import (ste_identity, ste_marginals, ste_marginals_single, spigot, sparsemap,
                            spigot_ce_argmax, spigot_eg_argmax)

class Decoder(torch.nn.Module):
    def __init__(self, embedding_dim, n_hid, p_drop, n_classes):
        super().__init__()

        self.compare = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_dim, n_hid),
            torch.nn.Dropout(p_drop),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hid, n_hid),
            torch.nn.Dropout(p_drop),
            torch.nn.ReLU())

        self.agg = torch.nn.Sequential(
            torch.nn.Linear(2 * n_hid, n_hid),
            torch.nn.Dropout(p_drop),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hid, n_classes))


    def forward(self, P_enc, H_enc, P_emb, H_emb, prem_mask, hypo_mask):
        # inter-attention scores (common)
        S = torch.einsum('ibh,jbh->bij', P_enc, H_enc)  # B x Np x Nh
        St = S.transpose(2, 1).clone()

        # inter-attention, hypo into premise
        S[prem_mask.T] = -np.inf
        P_attn = torch.softmax(S, dim=1)  # B x Np* x Nh
        H_ctx = torch.einsum('bij,ibh->jbh', P_attn, P_emb)
        H_ctx[hypo_mask] = 0  # remove padded rows
        # print('comparing:', H_emb.shape, H_ctx.shape)
        H_comp = self.compare(torch.cat([H_emb, H_ctx], dim=-1))

        # inter-attention, premise into hypo
        St[hypo_mask.T] = -np.inf
        H_attn = torch.softmax(St, dim=1)  # B x Nh* x Np
        P_ctx = torch.einsum('bji,jbh->ibh', H_attn, H_emb)
        P_ctx[prem_mask] = 0
        P_comp = self.compare(torch.cat([P_emb, P_ctx], dim=-1)) # G in the Parikh paper, section 3.2 

        # pool and predict
        pool = torch.cat([H_comp.sum(dim=0), P_comp.sum(dim=0)], dim=-1) # Section 3.3 in the Parikh paper
        out = self.agg(pool)

        return out


class Decomp(torch.nn.Module):
    def __init__(self, vectors, p_drop=.1, n_hid=50,
                 latent_type=None, latent_config=None,
                 n_classes=3, pad_ix=1):
        super().__init__()
        self.n_hid = n_hid
        self.p_drop = p_drop
        self.pad_ix = pad_ix

        self.embed = torch.nn.Embedding.from_pretrained(
            vectors,
            padding_idx=pad_ix,
            freeze=True)

        self.encode = torch.nn.Sequential(
            torch.nn.Linear(self.embed.embedding_dim, n_hid),
            torch.nn.Dropout(p_drop),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hid, n_hid),
            torch.nn.Dropout(p_drop),
            torch.nn.ReLU())

        self.decoder = Decoder(self.embed.embedding_dim, n_hid, p_drop, n_classes)

        self.latent_type = latent_type
        if self.latent_type:
            self.latent_config = latent_config
            self.latent_tree_gcn = LatentTreeGCN(n_hid, latent_type)


    def forward(self, batch, show_trees=False):
        prem = batch.premise
        hypo = batch.hypothesis

        if isinstance(prem, tuple):
            prem, _ = prem
            hypo, _ = hypo

        prem_mask = prem == self.pad_ix  # Np x B
        hypo_mask = hypo == self.pad_ix  # Nh x B

        P_emb = self.embed(prem)
        H_emb = self.embed(hypo)

        P_enc = self.encode(P_emb)  # Np x B x H
        H_enc = self.encode(H_emb)  # Nh x B x H

        # GCN representation of premise and hypothesis
        if self.latent_type:
            y = batch.label-1
            
            if show_trees:
                P_enc, H_enc, parser_output_p, parser_output_h = self.latent_tree_gcn(P_enc, H_enc, y, self.decoder, self.latent_config, (P_emb, H_emb, prem_mask, hypo_mask), show_trees)
            else:
                P_enc, H_enc = self.latent_tree_gcn(P_enc, H_enc, y, self.decoder, self.latent_config, (P_emb, H_emb, prem_mask, hypo_mask))

        # print('before decoder, in models.py:', P_enc.shape, H_enc.shape, P_emb.shape, H_emb.shape, prem_mask.shape, hypo_mask.shape)
        out = self.decoder(P_enc, H_enc, P_emb, H_emb, prem_mask, hypo_mask)

        if show_trees:
            return out, parser_output_p, parser_output_h
        else:
            return out


class LatentTreeGCN(torch.nn.Module):
    def __init__(self, hidden_size, latent_type):
        super().__init__()
        self.parser = DependencyParser(hidden_size)
        self.latent_type = latent_type
        self.attn_query = torch.nn.Linear(hidden_size*2, 1)

    def tree_representation(self, parser_output, parser_base_vectors):
        avg_heads = torch.einsum('bnm,bnh->bnh', parser_output, parser_base_vectors)
        tree_representation = torch.cat((parser_base_vectors, avg_heads), dim=2)
        
        return tree_representation

    def forward(self, inputs_p, inputs_h, y, decoder, latent_config, decoder_params, show_trees=False):
        # embedded = self.embeddings(inputs)
        # encoded, _ = self.encoder(embedded)
        latent_argmax=False
        latent_type = self.latent_type
        if latent_argmax:
            latent_type = 'argmax' # This is used for passing argmax on test time
        parser_output_p, parser_output_h = self.parser(inputs_p, inputs_h, y, latent_type, latent_config, decoder, decoder_params)

        # print('parser_output:', parser_output_p.shape, parser_output_h.shape)
        # Combine the parser result with the arc vectors to obtain a representation of the tree
        avg_heads_p = torch.einsum('bnm,bnh->bnh', parser_output_p, inputs_p)
        avg_heads_h = torch.einsum('bnm,bnh->bnh', parser_output_h, inputs_h)

        repr_p = torch.cat((inputs_p, avg_heads_p), dim=2) 
        repr_h = torch.cat((inputs_h, avg_heads_h), dim=2) 

        if show_trees:
            return repr_p, repr_h, parser_output_p, parser_output_h 
        else:
            return repr_p, repr_h


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

    def forward(self, X_p, X_h, y, latent_type, latent_config, decoder, decoder_params):
        """
        The possbile values for latent_type are:
        (# TODO: I need a better name for this variable!)
        - argmax - MAP on forward
        - marginals - MArginals on forward
        - ste_identity - MAP on forward, Identity on backward
        - ste_marginals - MAP on forward, marginals on backward
        - perturb_and_map - Gumbel noise + MAP on forward, marginals on backward
        """
        # print('decoder_params:', decoder_params)
        (P_emb, H_emb, mask_p, mask_h) = decoder_params

        arc_scores_p = self.calc_arc_scores(X_p)
        arc_scores_h = self.calc_arc_scores(X_h)

        # print('latent_type', latent_type)
        if False:
            parser_output = (arc_scores_p, arc_scores_h)
        elif latent_type == 'marginals':
            dist_p = DependencyCRF(arc_scores_p)
            parser_output_p = dist_p.marginals
            dist_h = DependencyCRF(arc_scores_h)
            parser_output_h = dist_h.marginals
            parser_output = (parser_output_p, parser_output_h)
        elif latent_type == 'ste_identity':
            parser_output = ste_identity(arc_scores_p, arc_scores_h, P_emb, H_emb, mask_p, mask_h, y, decoder, latent_config)
        elif latent_type == 'ste_marginals':
            parser_output = ste_marginals(arc_scores_p, arc_scores_h)
        elif latent_type == 'sparsemap':
            parser_output = sparsemap(arc_scores_p, arc_scores_h)
        elif latent_type == 'spigot':
            parser_output = spigot(arc_scores_p, arc_scores_h, P_emb, H_emb, mask_p, mask_h, y, decoder, latent_config)
        elif latent_type == 'spigot_ce_argmax':
            parser_output = spigot_ce_argmax(arc_scores_p, arc_scores_h, P_emb, H_emb, mask_p, mask_h, y, decoder, latent_config)
        elif latent_type == 'spigot_eg_argmax':
            parser_output = spigot_eg_argmax(arc_scores_p, arc_scores_h, P_emb, H_emb, mask_p, mask_h, y, decoder, latent_config)
        # elif latent_type == 'spigot_ce':
        #     parser_output = spigot_ce(arc_scores, decoder, X, y, latent_config)
        # elif latent_type == 'spigot_eg':
        #     parser_output = spigot_eg(arc_scores, decoder, X, y, latent_config)
        elif latent_type == 'perturb_and_map':
            # Perturn-and-MAP equals:
            # Add Gumbel noise to the arc scores and apply STE with marginals
            # m = Uniform(0, 1)
            # u = m.sample(arc_scores.shape)
            # g = -torch.log(-torch.log(u))
            # if device == 'cuda':
            #     g = g.cuda()
            # parser_output = ste_marginals(arc_scores+g)
            parser_output = (self.sample(arc_scores_p), self.sample(arc_scores_h))
        else: # assuming argmax as default
            # dist_p = DependencyCRF(arc_scores_p)
            # dist_h = DependencyCRF(arc_scores_h)
            parser_output = (argmax_batched(arc_scores_p), argmax_batched(arc_scores_h))

        return parser_output

    def sample(self, scores):
        m = Uniform(0, 1)
        u = m.sample(scores.shape)
        g = -torch.log(-torch.log(u))
        if device == 'cuda':
            g = g.cuda()
        return ste_marginals_single(scores+g)


