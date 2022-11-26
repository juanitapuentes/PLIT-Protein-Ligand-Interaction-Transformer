import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from model.model import DeeperGCN
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer, MLP, MM_AtomEncoder
import dgl
import dgllife
import numpy as np


from model.model_encoder import AtomEncoder, BondEncoder

import logging


class TransformerPAU(torch.nn.Module):
    def __init__(self, args, is_prot=False, saliency=False):
        super(TransformerPAU, self).__init__()

        # Set LM configuration
        self.molecule_gcn = DeeperGCN(args)
        self.num_layers = args.num_layers
        mlp_layers = args.mlp_layers
        hidden_channels = args.hidden_channels
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale
        self.conv_encode_edge = args.conv_encode_edge

        # Set overall model configuration
        self.dropout = args.dropout
        self.block = args.block
        self.add_virtual_node = args.add_virtual_node
        self.training = True
        self.args = args

        num_classes = args.nclasses
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.classification = nn.Linear(hidden_channels, num_classes)
        norm = args.norm

        # Print model parameters
        print(
            "The number of layers {}".format(self.num_layers),
            "Aggr aggregation method {}".format(aggr),
            "block: {}".format(self.block),
        )
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        elif self.block == "res":
            print("GraphConv->LN/BN->ReLU->Res")
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")
        elif self.block == "plain":
            print("GraphConv->LN/BN->ReLU")
        else:
            raise Exception("Unknown block Type")

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3, norm=norm))

        # Set GCN layer configuration
        for layer in range(self.num_layers):
            if conv == "gen":
                gcn = GENConv(
                    hidden_channels,
                    hidden_channels,
                    args,
                    aggr=aggr,
                    t=t,
                    learn_t=self.learn_t,
                    p=p,
                    learn_p=self.learn_p,
                    msg_norm=self.msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    encode_edge=self.conv_encode_edge,
                    bond_encoder=True,
                    norm=norm,
                    mlp_layers=mlp_layers,
                )
            else:
                raise Exception("Unknown Conv Type")
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # Set embbeding layers
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if saliency:
            self.atom_encoder = MM_AtomEncoder(emb_dim=hidden_channels)
        else:
            self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        self.device = torch.device("cuda:" + str(args.device))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dropout=0.08, activation="gelu", batch_first=True).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4).to(self.device)

        self.conv0 = torch.nn.Conv1d(256, 128, kernel_size=1).to(self.device)
        self.conv1 = torch.nn.Conv1d(128, 64, kernel_size=1).to(self.device)
        self.conv2 = torch.nn.Conv1d(64,32, kernel_size=1).to(self.device)
        self.conv3 = torch.nn.Conv1d(32, 1, kernel_size=1).to(self.device)

        self.graph_pred_linear = torch.nn.Linear(256, num_classes)


    def forward(self, input_batch, dropout=True, embeddings=False):

        max = 256
        h_prev = self.molecule_gcn(input_batch)

        pad_list = []
        mask_list = []

        if self.args.mode == "demo":
            bonds_idxs = input_batch.edge_index[0]
            edge_emb = self.bond_encoder(input_batch.edge_attr)
            bonds_per_atom = {}
            for kh in range(0, len(bonds_idxs)):
                if bonds_per_atom.get(int(bonds_idxs[kh])) == None:
                    bonds_per_atom[int(bonds_idxs[kh])] = edge_emb[kh, :]
                else:
                    prev = bonds_per_atom.get(int(bonds_idxs[kh]))
                    stack = torch.stack([prev, edge_emb[kh, :]], dim=0)
                    bonds_per_atom[int(bonds_idxs[kh])] = stack.mean(dim=0)

            bond_feat = torch.stack(list(bonds_per_atom.values()), 0).to(self.device)

            matrix = torch.cat([h_prev, bond_feat], dim=1)

            padded = torch.nn.functional.pad(matrix, (0, 0, 0, max - matrix.shape[0]))
            pad_list.append(padded)
            ones = torch.ones(matrix.shape)
            ones = torch.cat((ones, torch.zeros((max - matrix.shape[0], max))))

            for i in range(2):
                mask_list.append(ones)

        else:
            for ind in range(1, input_batch.ptr.shape[0]):
                bonds_idxs = input_batch[ind - 1].edge_index[0]
                edge_emb = self.bond_encoder(input_batch[ind - 1].edge_attr)

                bonds_per_atom = {}
                for kh in range(0, len(bonds_idxs)):
                    if bonds_per_atom.get(int(bonds_idxs[kh])) == None:
                        bonds_per_atom[int(bonds_idxs[kh])] = edge_emb[kh, :]
                    else:
                        prev = bonds_per_atom.get(int(bonds_idxs[kh]))
                        stack = torch.stack([prev, edge_emb[kh, :]], dim=0)
                        bonds_per_atom[int(bonds_idxs[kh])] = stack.mean(dim=0)

                bond_feat = torch.stack(list(bonds_per_atom.values()), 0).to(self.device)

                atoms = (input_batch.ptr[ind - 1], input_batch.ptr[ind])
                h_ = h_prev[atoms[0]:atoms[1]]
                matrix = torch.cat([h_, bond_feat], dim=1)

                padded = torch.nn.functional.pad(matrix, (0, 0, 0, max - matrix.shape[0]))
                pad_list.append(padded)
                ones = torch.ones(matrix.shape)
                ones = torch.cat((ones, torch.zeros((max - matrix.shape[0], max))))

                for i in range(2):
                    mask_list.append(ones)


        padded_batch = torch.stack(pad_list, 0).to(self.device)
        mask_batch = torch.stack(mask_list, 0).to(self.device)

        h_trans = self.transformer_encoder(padded_batch, mask_batch)
        h_trans = h_trans.view(h_trans.shape[0], h_trans.shape[2], h_trans.shape[1])

        conv = self.conv0(h_trans)
        conv = self.conv1(conv)
        conv = self.conv2(conv)
        conv = self.conv3(conv)

        conv = conv.view(1, conv.shape[0], conv.shape[2])[0]

        return self.graph_pred_linear(conv)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print("Final t {}".format(ts))
            else:
                logging.info("Epoch {}, t {}".format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print("Final p {}".format(ps))
            else:
                logging.info("Epoch {}, p {}".format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print("Final s {}".format(ss))
            else:
                logging.info("Epoch {}, s {}".format(epoch, ss))
