"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch

import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM,AutoModelForCausalLM
from model.structure_model import HGPSLPool
from torch_geometric.nn import global_mean_pool
# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs
from model.hetognn.net1 import GNNStack
from model.interaction import *
import torch




def compute_class_weights(targets, num_classes=2):
    # 计算每个类的样本数
    class_counts = torch.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    
    # 计算每个类的权重，权重可以是逆频率
    weights = total_samples / (num_classes * class_counts.float())
    
    return weights


opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]


class DynGraph(nn.Module):
    def __init__(self, feature_dim, k_intra=5, k_inter=5):
        super(DynGraph, self).__init__()
        self.theta1_intra = nn.Parameter(torch.randn(feature_dim))
        self.theta2_intra = nn.Parameter(torch.randn(feature_dim))
        self.theta_a_inter = nn.Parameter(torch.randn(feature_dim))
        self.theta_b_inter = nn.Parameter(torch.randn(feature_dim))
        self.sigma1 = nn.ReLU()
        self.sigma2 = nn.Sigmoid()
        self.k_intra = k_intra
        self.k_inter = k_inter

    def forward(self, tensor_a, tensor_b):
        Ua1 = tensor_a @ torch.diag(self.theta1_intra)
        Ua2 = tensor_a @ torch.diag(self.theta2_intra)
        A_intra_a = self.sigma1(
            self.sigma2(Ua1 @ Ua2.transpose(1, 2)) - self.sigma2(Ua2 @ Ua1.transpose(1, 2))
        )
        A_intra_a = self.sparsify(A_intra_a, self.k_intra)

        Ub1 = tensor_b @ torch.diag(self.theta1_intra)
        Ub2 = tensor_b @ torch.diag(self.theta2_intra)
        A_intra_b = self.sigma1(
            self.sigma2(Ub1 @ Ub2.transpose(1, 2)) - self.sigma2(Ub2 @ Ub1.transpose(1, 2))
        )
        A_intra_b = self.sparsify(A_intra_b, self.k_intra)

        Ua = tensor_a @ torch.diag(self.theta_a_inter)
        Ub = tensor_b @ torch.diag(self.theta_b_inter)
        A_inter = self.sigma1(self.sigma2(Ua @ Ub.transpose(1, 2)))
        A_inter = self.sparsify(A_inter, self.k_inter)

        return A_intra_a, A_intra_b, A_inter

    def sparsify(self, adjacency_matrix, k):
        batch_size, n, _ = adjacency_matrix.shape
        topk_values, topk_indices = torch.topk(adjacency_matrix, k=k, dim=-1)
        mask = torch.zeros_like(adjacency_matrix).scatter_(-1, topk_indices, 1)
        sparse_adjacency = adjacency_matrix * mask
        return sparse_adjacency


class HeterogeneousGraphNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads):
        super(HeterogeneousGraphNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.F1 = nn.ModuleList([nn.Linear(feature_dim, hidden_dim) for _ in range(num_heads)])
        self.F2 = nn.ModuleList([nn.Linear(feature_dim, hidden_dim) for _ in range(num_heads)])
        self.F3 = nn.ModuleList([nn.Linear(feature_dim, hidden_dim) for _ in range(num_heads)])

        self.W_intra_a = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_intra_b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_inter = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.W_output = nn.Linear(hidden_dim * num_heads, feature_dim)
        self.b_output = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, node_features_a, node_features_b, A_intra_a, A_intra_b, A_inter):
        updated_a = self.aggregate(node_features_a, [A_intra_a], self.W_intra_a)
        updated_b = self.aggregate(node_features_b, [A_intra_b], self.W_intra_b)

        inter_a = self.aggregate(node_features_a, [A_inter], self.W_inter, target_features=node_features_b)
        inter_b = self.aggregate(node_features_b, [A_inter.transpose(1, 2)], self.W_inter, target_features=node_features_a)

        updated_a += inter_a
        updated_b += inter_b

        return updated_a, updated_b

    def aggregate(self, source_features, adjacency_matrices, edge_weight, target_features=None):
        if target_features is None:
            target_features = source_features

        aggregated_features = torch.zeros_like(source_features)

        for adjacency_matrix in adjacency_matrices:
            attention_outputs = []
            message_outputs = []

            for i in range(self.num_heads):
                F1_u = self.F1[i](source_features)
                F2_v = self.F2[i](target_features)
                F3_u = self.F3[i](source_features)

                attention_scores = torch.matmul(F1_u, edge_weight @ F2_v.transpose(1, 2))

                attention_weights = F.softmax(attention_scores * adjacency_matrix, dim=-1)

                messages = torch.matmul(attention_weights, F3_u)

                attention_outputs.append(attention_weights)
                message_outputs.append(messages)

            attention_concat = torch.cat(attention_outputs, dim=-1)
            message_concat = torch.cat(message_outputs, dim=-1)

            aggregated_features += message_concat

        return F.relu(self.W_output(aggregated_features)) + self.b_output

class LearnableSortingNetwork(nn.Module):
    def __init__(self, feature_dim, epsilon=0.1, max_iters=50, eta=1e-3):
        """
        A Learnable Sorting Network with Sinkhorn Operator.
        
        Args:
            feature_dim (int): Dimension of the input features.
            epsilon (float): Control factor for the Sinkhorn operator.
            max_iters (int): Maximum number of Sinkhorn iterations.
            eta (float): Convergence threshold for the Sinkhorn operator.
        """
        super(LearnableSortingNetwork, self).__init__()
        self.feature_transform = nn.Linear(feature_dim, 1)  # Learnable parameter for each conformation
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.eta = eta

    def forward(self, h):
        """
        Forward pass of the Learnable Sorting Network.
        
        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, n, feature_dim),
                              where n is the number of conformers.

        Returns:
            torch.Tensor: Normalized conformer representation Z of shape (batch_size, n, feature_dim).
        """
        batch_size, n, feature_dim = h.shape

        # Compute sorting scores (S_a)
        scores = self.feature_transform(h).squeeze(-1)  # Shape: (batch_size, n)

        # Initial order (S_a) and sorted order (S_b)
        S_a = torch.arange(n, device=h.device).unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, n)
        S_b = torch.argsort(scores, dim=-1)  # Non-differentiable sorted order

        # Compute transport matrix Q
        S_a = S_a.unsqueeze(2).repeat(1, 1, n)  # Shape: (batch_size, n, n)
        S_b = S_b.unsqueeze(1).repeat(1, n, 1)  # Shape: (batch_size, n, n)
        Q = torch.exp(-torch.abs(S_a - S_b) / self.epsilon)  # Shape: (batch_size, n, n)

        # Sinkhorn iterations
        u = torch.ones(batch_size, n, device=h.device)  # Shape: (batch_size, n)
        v = torch.ones(batch_size, n, device=h.device)  # Shape: (batch_size, n)

        for _ in range(self.max_iters):
            v_new = 1.0 / (Q.transpose(1, 2).bmm(u.unsqueeze(-1)) + 1e-8)
            u_new = 1.0 / (Q.bmm(v_new) + 1e-8)
            v = v_new.squeeze(-1)
            u = u_new.squeeze(-1)

            # Convergence check
            delta = torch.abs((v * Q.transpose(1, 2).bmm(u.unsqueeze(-1)).squeeze(-1)) - (1.0 / n)).mean()
            if delta < self.eta:
                break

        # Normalize Q
        Q_normalized = torch.diag_embed(u) @ Q @ torch.diag_embed(v)  # Shape: (batch_size, n, n)

        # Compute final representation Z

        # print(Q_normalized.dtype)
        # print(h.dtype)
        h = h.to(torch.bfloat16)
        Q_normalized=Q_normalized.to(torch.bfloat16)

        Z = Q_normalized @ h  # Shape: (batch_size, n, feature_dim)

        return Z

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

import re
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list
    
class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune='freeze',
        peft_dir='',
        opt_model="facebook/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args
        ####是否使用子结构,子结构数
        self.zijiegou = args.zijiegou
        self.num_query_token=args.num_query_token
        self.autozijiegou = args.autozijiegou
        #####

        if args.use_3d:
            self.graph_encoder_3d, self.ln_graph_3d, self.dictionary = self.init_unimol_encoder(args)
        elif args.use_2d:
            self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio,self.zijiegou,self.num_query_token,self.autozijiegou,args.graph2d)

        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        if args.alignment=='qformer':
            self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
            ### remove the unused parameters
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        # #####################################
        ## initialize opt model

        opt_model = f"/home/cz/MolTC-main/backbone/{args.backbone}"
        
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})
        self.opt_tokenizer.add_tokens('<mol>') # molecule placeholder

        # self.opt_tokenizer.add_tokens('<start_property>')
        # self.opt_tokenizer.add_tokens('<end_property>')

        # if opt_model == "./deepseek/deepseek-1.5b":
        self.opt_tokenizer.add_special_tokens({
    'additional_special_tokens': ['[START_I_SMILES]', '[END_I_SMILES]','</s>']
        })

        self.opt_tokenizer.add_special_tokens({
    'additional_special_tokens': ['<start_property>', '<end_property>','<GraEmb1>','<GraEmb2>']
        })
        # self.opt_tokenizer.add_tokens('<GraEmb1>')
        # self.opt_tokenizer.add_tokens('<GraEmb2>')

        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        self.opt_tokenizer.start_property_token_id = self.opt_tokenizer("<start_property>", add_special_tokens=False).input_ids[0]
        self.opt_tokenizer.end_property_token_id = self.opt_tokenizer("<end_property>", add_special_tokens=False).input_ids[0]
        self.opt_tokenizer.GraEmb1_token_id = self.opt_tokenizer("<GraEmb1>", add_special_tokens=False).input_ids[0]
        self.opt_tokenizer.GraEmb2_token_id = self.opt_tokenizer("<GraEmb2>", add_special_tokens=False).input_ids[0]
        
        self.pad_token_id = self.opt_tokenizer.pad_token_id
        
        
        self.collater = Collater([], [])
        
        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model)
        else:
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16
            print(f"start loading {opt_model}")
            # if opt_model == "./deepseek/deepseek-1.5b":
            self.opt_model = AutoModelForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
            # else:
            #     self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)

        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer)) ## this will cause bug when full fine-tuning the opt model

        self.llm_tune = llm_tune
        if llm_tune == 'lora':
            if peft_dir:
                print('*****'*10)
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
                #for name, param in self.opt_model.named_parameters():
                #    param.requires_grad = False
            else:
                if self.args.peft_config:
                    print('*****22222'*10)
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    print('*****3333333'*10)
                    if opt_model == "./deepseek/deepseek-1.5b":
                        print('*****4444444'*10)
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM, 
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                            inference_mode=False, 
                            r=args.lora_r, 
                            lora_alpha=args.lora_alpha, 
                            lora_dropout=args.lora_dropout
                        )
                    else:
                        print('*****5555555'*10)
                        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
                        # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
                        # "query_key_value"
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
            self.opt_model.print_trainable_parameters()
            
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        if args.alignment=='qformer':
            self.opt_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
            )
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)  
        
        self.dyngraph = DynGraph(512)
        self.Heg = HeterogeneousGraphNetwork(512,128,4)

        self.MLP = nn.Sequential(
            # nn.Linear(256,512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.opt_model.config.hidden_size)
        )
        
        self.bilinear_attn = BilinearAttention(self.opt_model.config.hidden_size)
        self.self_attn = SelfAttention(self.opt_model.config.hidden_size)
        self.cross_attn = CrossAttention(self.opt_model.config.hidden_size)
        self.highway = Highway(self.opt_model.config.hidden_size)
        self.gated_fusion = GatedFusion(self.opt_model.config.hidden_size)
        self.bilinear_fusion = BilinearFusion(self.opt_model.config.hidden_size)


        self.conMLP = nn.Sequential(
            # nn.Linear(256,512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.opt_model.config.hidden_size)
        )
        
        self.graemb1m = GNNStack(gnn_model_type='dyGIN2d', num_layers=3, 
                    groups=1, pool_ratio=0.2, kern_size=[9, 5, 3], 
                    in_dim=64, hidden_dim=128, out_dim=256, 
                    seq_len=0, num_nodes=self.args.num_query_token, num_classes=0)
        
        self.graemb2m = GNNStack(gnn_model_type='dyGIN2d', num_layers=3, 
                    groups=1, pool_ratio=0.2, kern_size=[9, 5, 3], 
                    in_dim=64, hidden_dim=128, out_dim=256, 
                    seq_len=0, num_nodes=self.args.num_query_token, num_classes=0)
        
        self.hetographm = GNNStack(gnn_model_type='dyGIN2d', num_layers=3, 
                    groups=1, pool_ratio=0.2, kern_size=[9, 5, 3], 
                    in_dim=64, hidden_dim=128, out_dim=256, 
                    seq_len=0, num_nodes=2*self.args.num_query_token, num_classes=0)

        self.lsort = LearnableSortingNetwork(feature_dim=512)

    def forward(self, batch):
        if self.args.use_3d:
            graph1, graph2,prompt_tokens, text_tokens= batch
            embeds_list = []
            for i in range(len(graph1)):
                graph1_embeds, graph1_masks= self.graph_encoder_3d(*graph1[i])
                # print('graph1_embeds',graph1_embeds.shape)
                if graph1_masks is not None:
                    graph1_masks = ~graph1_masks
                    # print('graph1_embeds******',graph1_embeds.shape)
                    graph1_embeds = graph1_embeds[graph1_masks].reshape(20,-1,512)
                else:
                    graph1_embeds=graph1_embeds.reshape(20,-1,512)
                #graph1_embeds数据格式 numconformers*原子数*512
                graph1_embeds = self.ln_graph_3d(graph1_embeds)

                graph1_embeds = torch.mean(graph1_embeds[:self.num_query_token],dim=1)
                embeds_list.append(graph1_embeds)
            #batchsize*numquery*512
            graphconformer1 = torch.stack(embeds_list,dim=0)

            # print('111111111*****',graphconformer1.shape)
            # graphconformer1 = self.graemb1m(graphconformer1)
            

            embeds_list = []
            for i in range(len(graph2)):
                graph2_embeds, graph2_masks= self.graph_encoder_3d(*graph2[i])
                # print('graph2_embeds',graph2_embeds.shape)
                if graph2_masks is not None:
                    graph2_masks = ~graph2_masks
                    graph2_embeds = graph2_embeds[graph2_masks].reshape(20,-1,512)
                else:
                    graph2_embeds = graph2_embeds.reshape(20,-1,512)
                graph2_embeds = self.ln_graph_3d(graph2_embeds)
                graph2_embeds = torch.mean(graph2_embeds[:self.num_query_token],dim=1)
                embeds_list.append(graph2_embeds)

            graphconformer2 = torch.stack(embeds_list,dim=0)
            # graphconformer2 = self.graemb2m(graphconformer2)

            # print(graphconformer1.shape,graphconformer2.shape)

            A_intra_a, A_intra_b, A_inter = self.dyngraph(graphconformer1,graphconformer2)

            # aggregated_features1=torch.cat((graphconformer1, graphconformer2), dim=1).unsqueeze(1)

            # aggregated_features1 = self.hetographm(aggregated_features1,True)

            
            # mol_tokens = self.MLP(aggregated_features1)

            graphconformer1, graphconformer2 = self.Heg(graphconformer1,graphconformer2,A_intra_a, A_intra_b, A_inter)

            device = graphconformer2.device

            empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)

            
            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            #文本编码
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

            # prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

            # graphconformer1 = self.lsort(graphconformer1)
            # graphconformer2 = self.lsort(graphconformer2)

            if self.args.alignment == 'mlp':
                #batch*20*512
                graphconformer1 = self.conMLP(graphconformer1)
                graphconformer2 = self.conMLP(graphconformer2)
            elif self.args.alignment == 'qformer':
                mask = torch.ones(self.args.batch_size, 20, dtype=torch.bool)
                query_tokens1 = self.query_tokens.expand(graphconformer1.shape[0], -1, -1)
                query_output1 = self.Qformer.bert(
                    query_embeds=query_tokens1,
                    encoder_hidden_states=graphconformer1,
                    encoder_attention_mask=mask, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                graphconformer1 = self.opt_proj(query_output1.last_hidden_state)

                query_tokens2 = self.query_tokens.expand(graphconformer2.shape[0], -1, -1)
                query_output2 = self.Qformer.bert(
                    query_embeds=query_tokens2,
                    encoder_hidden_states=graphconformer2,
                    encoder_attention_mask=mask, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                graphconformer2 = self.opt_proj(query_output2.last_hidden_state)
                

            if self.args.use_inter:
                if self.args.interaction=='mean':
                    inter = (graphconformer1 + graphconformer2) / 2
                elif self.args.interaction=='BilinearFusion':
                    inter = self.bilinear_fusion(graphconformer1, graphconformer2)
                elif self.args.interaction=='SelfAttention':
                    sa1 = self.self_attn(graphconformer1)
                    sa2 = self.self_attn(graphconformer2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='CrossAttention':
                    inter = self.cross_attn(graphconformer1, graphconformer2)
                elif self.args.interaction=='Highway':
                    sa1 = self.Highway(graphconformer1)
                    sa2 = self.Highway(graphconformer2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='GatedFusion':
                    inter = self.gated_fusion(graphconformer1, graphconformer2)
                elif self.args.interaction=='BilinearAttention':
                    inter = self.bilinear_attn(graphconformer1, graphconformer2)
                prompt_embeds[prompt_tokens.is_mol_token] = inter.flatten(0, 1)


            prompt_embeds[prompt_tokens.is_GraEmb1_token] = graphconformer1.flatten(0, 1)
            prompt_embeds[prompt_tokens.is_GraEmb2_token] = graphconformer2.flatten(0, 1)




            inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
            inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)


            attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
            
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                # output_attentions=True,
                output_hidden_states=True,
                labels=targets,
            )
            # return outputs

            loss = outputs.loss

            return {"loss": loss}
        
        elif self.args.use_2d:
            graphs1, graphs2,prompt_tokens, text_tokens= batch
            #print(graphs)pyg图数据格式
            #print(prompt_tokens)下面两个都是64 54 38这种
            #print(text_tokens)
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
            if not self.tune_gnn:
                graph_embeds1 = graph_embeds1.detach()
            graph_embeds1 = self.ln_graph(graph_embeds1, graph_masks1)
            #print(graph_embeds)应该是读出完之后的向量了
            device = graph_embeds1.device

            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
            if not self.tune_gnn:
                graph_embeds2 = graph_embeds1.detach()
            graph_embeds2 = self.ln_graph(graph_embeds2, graph_masks2)

            if self.args.alignment == 'mlp':
                target_num = self.args.num_query_token
                current_num = graph_embeds1.size(1)

                if current_num >= target_num:
                    mol_tokens1 = graph_embeds1[:, :target_num, :]
                    mol_tokens2 = graph_embeds2[:, :target_num, :]
                else:
                    pad_size = target_num - current_num
                    pad_tensor1 = torch.zeros(self.args.batch_size, pad_size, 512, device=graph_embeds1.device, dtype=graph_embeds1.dtype)
                    mol_tokens1 = torch.cat([graph_embeds1, pad_tensor1], dim=1)

                    pad_tensor2 = torch.zeros(self.args.batch_size, pad_size, 512, device=graph_embeds2.device, dtype=graph_embeds2.dtype)
                    mol_tokens2 = torch.cat([graph_embeds2, pad_tensor2], dim=1)

                mol_tokens1 = self.conMLP(mol_tokens1)
                mol_tokens2 = self.conMLP(mol_tokens2)



            elif self.args.alignment == 'qformer':
                query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
                #print(query_tokens1.size())#也是向量罢了
                query_output1 = self.Qformer.bert(
                    query_embeds=query_tokens1,
                    encoder_hidden_states=graph_embeds1,
                    encoder_attention_mask=graph_masks1, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                #print(query_output1.size())
                mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
                #print(mol_tokens1.size())
                #这里应该是去取出了分子的向量,并不是token，这里是将768层 变换称了2048层
                
                
            
                #print(graph_embeds)应该是读出完之后的向量了
                device = graph_embeds2.device
                query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
                #print(query_tokens)也是向量罢了
                query_output2 = self.Qformer.bert(
                    query_embeds=query_tokens2,
                    encoder_hidden_states=graph_embeds2,
                    encoder_attention_mask=graph_masks2, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)
            

            
            
            
            # mol_tokens = torch.cat([mol_tokens1,mol_tokens2], dim=1)
            empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)


            if self.args.use_inter:
                if self.args.interaction=='mean':
                    inter = (mol_tokens1 + mol_tokens2) / 2
                elif self.args.interaction=='BilinearFusion':
                    inter = self.bilinear_fusion(mol_tokens1, mol_tokens2)
                elif self.args.interaction=='SelfAttention':
                    sa1 = self.self_attn(mol_tokens1)
                    sa2 = self.self_attn(mol_tokens2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='CrossAttention':
                    inter = self.cross_attn(mol_tokens1, mol_tokens2)
                elif self.args.interaction=='Highway':
                    sa1 = self.Highway(mol_tokens1)
                    sa2 = self.Highway(mol_tokens2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='GatedFusion':
                    inter = self.gated_fusion(mol_tokens1, mol_tokens2)
                elif self.args.interaction=='BilinearAttention':
                    inter = self.bilinear_attn(mol_tokens1, mol_tokens2)
            prompt_embeds[prompt_tokens.is_mol_token] = inter.flatten(0, 1)

            prompt_embeds[prompt_tokens.is_GraEmb1_token] = mol_tokens1.flatten(0, 1)
            prompt_embeds[prompt_tokens.is_GraEmb2_token] = mol_tokens2.flatten(0, 1)
            inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
            inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
            attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
            
            
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if self.args.use_3d:
            graph1 = samples['graphs1']
            prompt_tokens = samples['prompt_tokens']
            graph2 = samples['graphs2']

            embeds_list = []

            for i in range(len(graph1)):
                graph1_embeds, graph1_masks= self.graph_encoder_3d(*graph1[i])
                if graph1_masks is not None:
                    graph1_masks = ~graph1_masks
                    graph1_embeds = graph1_embeds[graph1_masks].reshape(20,-1,512)
                else:
                    graph1_embeds=graph1_embeds.reshape(20,-1,512)
                graph1_embeds = self.ln_graph_3d(graph1_embeds)
                graph1_embeds = torch.mean(graph1_embeds[:self.num_query_token],dim=1)
                embeds_list.append(graph1_embeds)

            graphconformer1 = torch.stack(embeds_list,dim=0)
            # graphconformer1 = self.graemb1m(graphconformer1)

            # print('gengraphconformer1',graphconformer1.shape)

            embeds_list = []
            for i in range(len(graph2)):
                graph2_embeds, graph2_masks= self.graph_encoder_3d(*graph2[i])
                if graph2_masks is not None:
                    graph2_masks = ~graph2_masks
                    graph2_embeds = graph2_embeds[graph2_masks].reshape(20,-1,512)
                else:
                    graph2_embeds = graph2_embeds.reshape(20,-1,512)
                graph2_embeds = self.ln_graph_3d(graph2_embeds)
                graph2_embeds = torch.mean(graph2_embeds[:self.num_query_token],dim=1)
                embeds_list.append(graph2_embeds)
            graphconformer2 = torch.stack(embeds_list,dim=0)
            # graphconformer2 = self.graemb2m(graphconformer2)
            
            A_intra_a, A_intra_b, A_inter = self.dyngraph(graphconformer1,graphconformer2)
            graphconformer1, graphconformer2 = self.Heg(graphconformer1,graphconformer2,A_intra_a, A_intra_b, A_inter)

            # aggregated_features1=torch.cat((graphconformer1, graphconformer2), dim=1).unsqueeze(1)
            # aggregated_features1 = self.hetographm(aggregated_features1,True)
            # # print('aggregated_features1',aggregated_features1.shape)
            # mol_tokens = self.MLP(aggregated_features1)

            
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

            # prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

            # graphconformer1 = self.lsort(graphconformer1)
            # graphconformer2 = self.lsort(graphconformer2)
            if self.args.alignment == 'mlp':
                #batch*20*512
                graphconformer1 = self.conMLP(graphconformer1)
                graphconformer2 = self.conMLP(graphconformer2)
            elif self.args.alignment == 'qformer':
                mask = torch.ones(self.args.inference_batch_size, 20, dtype=torch.bool)
                query_tokens1 = self.query_tokens.expand(graphconformer1.shape[0], -1, -1)
                query_output1 = self.Qformer.bert(
                    query_embeds=query_tokens1,
                    encoder_hidden_states=graphconformer1,
                    encoder_attention_mask=mask, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                graphconformer1 = self.opt_proj(query_output1.last_hidden_state)

                query_tokens2 = self.query_tokens.expand(graphconformer2.shape[0], -1, -1)
                query_output2 = self.Qformer.bert(
                    query_embeds=query_tokens2,
                    encoder_hidden_states=graphconformer2,
                    encoder_attention_mask=mask, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                graphconformer2 = self.opt_proj(query_output2.last_hidden_state)

            if self.args.use_inter:
                if self.args.interaction=='mean':
                    inter = (graphconformer1 + graphconformer2) / 2
                elif self.args.interaction=='BilinearFusion':
                    inter = self.bilinear_fusion(graphconformer1, graphconformer2)
                elif self.args.interaction=='SelfAttention':
                    sa1 = self.self_attn(graphconformer1)
                    sa2 = self.self_attn(graphconformer2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='CrossAttention':
                    inter = self.cross_attn(graphconformer1, graphconformer2)
                elif self.args.interaction=='Highway':
                    sa1 = self.Highway(graphconformer1)
                    sa2 = self.Highway(graphconformer2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='GatedFusion':
                    inter = self.gated_fusion(graphconformer1, graphconformer2)
                elif self.args.interaction=='BilinearAttention':
                    inter = self.bilinear_attn(graphconformer1, graphconformer2)
                prompt_embeds[prompt_tokens.is_mol_token] = inter.flatten(0, 1)

            prompt_embeds[prompt_tokens.is_GraEmb1_token] = graphconformer1.flatten(0, 1)
            prompt_embeds[prompt_tokens.is_GraEmb2_token] = graphconformer2.flatten(0, 1)
            

            # outputs = self.opt_model.generate(
            #     input_ids=prompt_embeds,
            #     attention_mask=prompt_tokens.attention_mask,
            #     do_sample=True,
            #     top_p=top_p,
            #     top_k=0,

            #     max_new_tokens=50)
            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,

                output_hidden_states=True,
                return_dict_in_generate=True
                # use_cache=False,
            )

            output_text = self.opt_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            # print('**************',output_text)
            return output_text
        elif self.args.use_2d:
            graphs1 = samples['graphs1']
            prompt_tokens = samples['prompt_tokens']
            # prompt_lens = samples['prompt_lens']
            # with self.maybe_autocast():
            graph_embeds1, graph_masks1 = self.graph_encoder(graphs1)
            graph_embeds1 = self.ln_graph(graph_embeds1)

            graphs2 = samples['graphs2']
            # prompt_lens = samples['prompt_lens']
            # with self.maybe_autocast():
            graph_embeds2, graph_masks2 = self.graph_encoder(graphs2)
            graph_embeds2 = self.ln_graph(graph_embeds2)


            if self.args.alignment == 'mlp':
                target_num = self.args.num_query_token
                current_num = graph_embeds1.size(1)

                if current_num >= target_num:
                    mol_tokens1 = graph_embeds1[:, :target_num, :]
                    mol_tokens2 = graph_embeds2[:, :target_num, :]
                else:
                    pad_size = target_num - current_num
                    pad_tensor1 = torch.zeros(self.args.inference_batch_size, pad_size, 512, device=graph_embeds1.device, dtype=graph_embeds1.dtype)
                    mol_tokens1 = torch.cat([graph_embeds1, pad_tensor1], dim=1)

                    pad_tensor2 = torch.zeros(self.args.inference_batch_size, pad_size, 512, device=graph_embeds2.device, dtype=graph_embeds2.dtype)
                    mol_tokens2 = torch.cat([graph_embeds2, pad_tensor2], dim=1)

                mol_tokens1 = self.conMLP(mol_tokens1)
                mol_tokens2 = self.conMLP(mol_tokens2)



            elif self.args.alignment == 'qformer':
                query_tokens1 = self.query_tokens.expand(graph_embeds1.shape[0], -1, -1)
                #print(query_tokens1.size())#也是向量罢了
                query_output1 = self.Qformer.bert(
                    query_embeds=query_tokens1,
                    encoder_hidden_states=graph_embeds1,
                    encoder_attention_mask=graph_masks1, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                #print(query_output1.size())
                mol_tokens1 = self.opt_proj(query_output1.last_hidden_state)
                #print(mol_tokens1.size())
                #这里应该是去取出了分子的向量,并不是token，这里是将768层 变换称了2048层
                
                
            
                #print(graph_embeds)应该是读出完之后的向量了
                device = graph_embeds2.device
                query_tokens2 = self.query_tokens.expand(graph_embeds2.shape[0], -1, -1)
                #print(query_tokens)也是向量罢了
                query_output2 = self.Qformer.bert(
                    query_embeds=query_tokens2,
                    encoder_hidden_states=graph_embeds2,
                    encoder_attention_mask=graph_masks2, # fixme: check whether this mask is correct
                    return_dict=True,
                )
                mol_tokens2 = self.opt_proj(query_output2.last_hidden_state)


            # mol_tokens=torch.cat([mol_tokens1,mol_tokens2],dim=1)
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

            if self.args.use_inter:
                if self.args.interaction=='mean':
                    inter = (mol_tokens1 + mol_tokens2) / 2
                elif self.args.interaction=='BilinearFusion':
                    inter = self.bilinear_fusion(mol_tokens1, mol_tokens2)
                elif self.args.interaction=='SelfAttention':
                    sa1 = self.self_attn(mol_tokens1)
                    sa2 = self.self_attn(mol_tokens2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='CrossAttention':
                    inter = self.cross_attn(mol_tokens1, mol_tokens2)
                elif self.args.interaction=='Highway':
                    sa1 = self.Highway(mol_tokens1)
                    sa2 = self.Highway(mol_tokens2)
                    inter = (sa1+sa2)/2
                elif self.args.interaction=='GatedFusion':
                    inter = self.gated_fusion(mol_tokens1, mol_tokens2)
                elif self.args.interaction=='BilinearAttention':
                    inter = self.bilinear_attn(mol_tokens1, mol_tokens2)


            prompt_embeds[prompt_tokens.is_GraEmb1_token] = mol_tokens1.flatten(0, 1)
            prompt_embeds[prompt_tokens.is_GraEmb2_token] = mol_tokens2.flatten(0, 1)

            outputs = self.opt_model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_tokens.attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                # use_cache=False,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            output_text = [text.strip() for text in output_text]
            return output_text


