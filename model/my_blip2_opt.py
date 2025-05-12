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


        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio,self.zijiegou,self.num_query_token,self.autozijiegou)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        
        # self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        # ### remove the unused parameters
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # #####################################
        ## initialize opt model

        # opt_model = "./galactica-1.3b"
        opt_model = "./backbone/galactica-1.3b"
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>', 'sep_token': '</s>'})
        self.opt_tokenizer.add_tokens('<mol>') # molecule placeholder

        # self.opt_tokenizer.add_tokens('<start_property>')
        # self.opt_tokenizer.add_tokens('<end_property>')
        if opt_model == "./deepseek/falcon/7b":
                    self.opt_tokenizer.add_special_tokens({
    'additional_special_tokens': ['[START_I_SMILES]', '[END_I_SMILES]','</s>']
        })
                    
        self.opt_tokenizer.add_special_tokens({
    'additional_special_tokens': ['<start_property>', '<end_property>','<sub1>','<sub2>']
        })
        self.opt_tokenizer.add_tokens('<GraEmb1>')
        self.opt_tokenizer.add_tokens('<GraEmb2>')

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
            # self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
            self.opt_model = AutoModelForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
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
                        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

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

        # self.opt_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        # )
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self.MLP = nn.Sequential(
            nn.Linear(args.gin_hidden_dim, args.bert_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.bert_hidden_dim, self.opt_model.config.hidden_size)
            
            # nn.Linear(args.gin_hidden_dim, self.opt_model.config.hidden_size)
        )
        # self.num_query_token = 8
        if self.autozijiegou==True:
            # self.zijiegoumodel = HGPSLPool(args.gin_hidden_dim, 8, True, True, True, 0.5)
            self.zijiegoumodel = HGPSLPool(args.gin_hidden_dim, self.num_query_token, True, True, True, 0.5)
        self.pool = global_mean_pool

        self.mlptoken = nn.Sequential(
            nn.Linear(self.opt_model.config.hidden_size, args.gin_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gin_hidden_dim, 2)
        )

        self.mlpzijiegou = nn.Sequential(
            nn.Linear(self.opt_model.config.hidden_size, args.gin_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gin_hidden_dim, 37)
        )

        self.mlptoken_solve = nn.Sequential(
            nn.Linear(self.opt_model.config.hidden_size, args.gin_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gin_hidden_dim, 1)
        )
        

        self.attenlinear = nn.Linear(args.gin_hidden_dim,1)


    def forward(self, batch):
        if self.args.double==False and self.args.solve == False and self.args.fangguangtuan == False and self.args.DDI == False:
            graphs, prompt_tokens, text_tokens = batch
            #print(graphs)pyg图数据格式
            #print(prompt_tokens)下面两个都是64 54 38这种
            #print(text_tokens)
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            if not self.tune_gnn:
                graph_embeds = graph_embeds.detach()
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            # 
            
            #print(graph_embeds)应该是读出完之后的向量了
            device = graph_embeds.device
            #可学习变量
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            #print(query_tokens)也是向量罢了
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state)
            #print(mol_tokens)
            #这里应该是去取出了分子的向量,并不是token
            empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            #处理后的分子进行编码 文本信息
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
            #图信息
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)
            #text_tokens为他们是否有交互性质
            inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
            #拼接
            inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
            attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
            #print(targets)
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            return {"loss": loss}
        else:
            if self.autozijiegou==False:
                graphs1, graphs2,zijiegou1,zijiegou2,prompt_tokens, text_tokens= batch
    
                zijiegou_embeds1= self.graph_encoder(graphs1,zijiegou1)
            else:
                graph1, graph2,prompt_tokens, text_tokens= batch
                node1rep,graph1pre= self.graph_encoder(graph1)

                edge_index1 = graph1.edge_index
                edge_attr1 = graph1.edge_attr
                batch1 = graph1.batch
                newx = node1rep
                newedgeindex = edge_index1
                newedge = edge_attr1
                newbatch = batch1

                # zijiegoutol_final1 = graph1pre
                newx, newedgeindex, newedge, newbatch = self.zijiegoumodel(newx, newedgeindex, newedge, newbatch)
                zijiegoutol_final1 = newx.reshape(-1,self.num_query_token,self.args.gin_hidden_dim)

                graphembed1 = self.MLP(zijiegoutol_final1)
            
            if not self.tune_gnn:
                zijiegoutol_final1 = zijiegoutol_final1.detach()
            device = zijiegoutol_final1.device

            # mol_tokens1 = self.MLP(zijiegoutol_final1)
            # print('dddddd',mol_tokens1.shape)

            if self.autozijiegou==False:
                zijiegou_embeds2= self.graph_encoder(graphs2,zijiegou2)
            else:
                node2rep,graph2pre= self.graph_encoder(graph2)

                edge_index2 = graph2.edge_index
                edge_attr2 = graph2.edge_attr
                batch2 = graph2.batch
                newx2 = node2rep
                newedgeindex2 = edge_index2
                newedge2 = edge_attr2
                newbatch2 = batch2

                newx2, newedgeindex2, newedge2, newbatch2 = self.zijiegoumodel(newx2, newedgeindex2, newedge2, newbatch2)
                zijiegoutol_final2 = newx2.reshape(-1,self.num_query_token,self.args.gin_hidden_dim)

                # zijiegoutol_final2 = graph2pre

                graphembed2 = self.MLP(zijiegoutol_final2)

            zijiegou_final = (zijiegoutol_final1.unsqueeze(2)+zijiegoutol_final2.unsqueeze(1))/2
            zijiegou_final = zijiegou_final.reshape(-1,(self.num_query_token)*(self.num_query_token),self.args.gin_hidden_dim)
            attenscore = self.attenlinear(zijiegou_final)
            attenscore = attenscore.squeeze(-1)
            perm = torch.argsort(attenscore, dim=1, descending=True)
            top_indices = perm[:, :self.num_query_token]
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, zijiegou_final.size(-1))
            aggregated_features1 = torch.gather(zijiegou_final, dim=1, index=top_indices_expanded)

            if not self.tune_gnn:
                zijiegoutol_final2 = zijiegoutol_final2.detach()       
            device = zijiegoutol_final2.device


            mol_tokens = self.MLP(aggregated_features1)

            empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)

            
            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

            # prompt_embeds[prompt_tokens.is_GraEmb1_token] = graphembed1
            # prompt_embeds[prompt_tokens.is_GraEmb2_token] = graphembed2


            prompt_embeds[prompt_tokens.is_GraEmb1_token] = graphembed1.flatten(0, 1)
            prompt_embeds[prompt_tokens.is_GraEmb2_token] = graphembed2.flatten(0, 1)

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
            #分类任务mlp
            # if False:
            if self.args.solve == False:
            # if False:
                # pass
                criterionzijiegou = nn.BCEWithLogitsLoss()
                loss_zijiegou = 0

                zijiegoupos1 = prompt_embeds.shape[1]+8
                zijiegoutoken1 = outputs.hidden_states[-1][:,zijiegoupos1,:]
                zijiegoutoken1 = self.mlpzijiegou(zijiegoutoken1)


                labels = graph1.zijiegouy.view(self.args.batch_size,37)

                loss_zijiegou += criterionzijiegou(zijiegoutoken1,labels.float())

                zijiegoupos2 = prompt_embeds.shape[1]+19
                zijiegoutoken2 = outputs.hidden_states[-1][:,zijiegoupos2,:]
                zijiegoutoken2 = self.mlpzijiegou(zijiegoutoken2)

                labels = graph2.zijiegouy.view(self.args.batch_size,37)

                loss_zijiegou += criterionzijiegou(zijiegoutoken2,labels.float())

            if self.args.solve==False:
            # if False:
                # pass
                posclassfied = prompt_embeds.shape[1]+26
                specific_token = outputs.hidden_states[-1][:,posclassfied,:]
                token_mlp = self.mlptoken(specific_token)
                token_probs = F.softmax(token_mlp, dim=1)
                predicted_classes = torch.argmax(token_probs, dim=1)
                criterion = nn.CrossEntropyLoss()
                targets = graph1.y
                
                class_weights = compute_class_weights(targets, num_classes=2)
                criterion.weight = class_weights
                
                loss2 = criterion(token_mlp, graph1.y)
            else :
                # posclassfiedstart = prompt_embeds.shape[1]+5
                # posclassfiedend = prompt_embeds.shape[1]+10
                # specific_token = outputs.hidden_states[-1][:,posclassfiedstart:posclassfiedend,:]
                # mean_token = specific_token.mean(dim=1) 
                mean_token = outputs.hidden_states[-1][:,6,:]
                token_mlp = self.mlptoken_solve(mean_token)
                # print('token_mlp'*4,token_mlp.shape)
                criterion = nn.MSELoss()
                loss2 = criterion(token_mlp, graph1.y)
                # print('loss2'*5,loss2)
            if self.args.solve == False:
                loss = outputs.loss+loss2+loss_zijiegou
                # loss = outputs.loss
            else:
                loss = outputs.loss+loss2


            return {"loss": loss}

    def forward_reaction(self, batch):
        # print('&&&&&&&&&&&&&&&&&&&&&&')
        reaction_tokens, notes_tokens, graphs = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]

        if False:
            if self.llm_tune:
                react_embeds = self.opt_model.model.get_decoder().embed_tokens(reaction_tokens.input_ids) # shape = [B, max_len, D]
                notes_embeds = self.opt_model.model.get_decoder().embed_tokens(notes_tokens.input_ids)
            else:
                react_embeds = self.opt_model.model.decoder.embed_tokens(reaction_tokens.input_ids) # shape = [B, max_len, D]
                notes_embeds = self.opt_model.model.decoder.embed_tokens(notes_tokens.input_ids) # shape = [B, max_len, D]
        else:
            react_embeds = self.opt_model.get_input_embeddings()(reaction_tokens.input_ids)
            notes_embeds = self.opt_model.get_input_embeddings()(notes_tokens.input_ids)

        react_embeds[reaction_tokens.is_ph_token] = mol_tokens.flatten(0, 1)
        inputs_embeds = torch.cat((react_embeds, notes_embeds), dim=1)

        targets = notes_tokens.input_ids.masked_fill(
            notes_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(reaction_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        attention_mask = torch.cat([reaction_tokens.attention_mask, notes_tokens.attention_mask], dim=1)

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
        if self.args.double == True or self.args.solve == True or self.args.fangguangtuan == True or self.args.DDI == True:
            if self.autozijiegou==True:
                # print('&&&&&&&&&&&&&&&&&&&')
                graph1 = samples['graphs1']
                # zijiegou1 = samples['zijiegou1']
                prompt_tokens = samples['prompt_tokens']
                graph2 = samples['graphs2']
                # zijiegou2 = samples['zijiegou2']
                    
                node1rep,graph1pre= self.graph_encoder(graph1)
                # print('qqqqqqqq1')
                # print('11111',graph1)
                edge_index1 = graph1.edge_index
                edge_attr1 = graph1.edge_attr
                batch1 = graph1.batch
                # print('qqqqqqq2')
                newx = node1rep
                newedgeindex = edge_index1
                newedge = edge_attr1
                newbatch = batch1

                # zijiegoutol_final1 = graph1pre
                newx, newedgeindex, newedge, newbatch = self.zijiegoumodel(newx, newedgeindex, newedge, newbatch)
                zijiegoutol_final1 = newx.reshape(-1,self.num_query_token,self.args.gin_hidden_dim)
                graphembed1 = self.MLP(zijiegoutol_final1)
                
                node2rep,graph2pre= self.graph_encoder(graph2)
                edge_index2 = graph2.edge_index
                edge_attr2 = graph2.edge_attr
                batch2 = graph2.batch
                newx2 = node2rep
                newedgeindex2 = edge_index2
                newedge2 = edge_attr2
                newbatch2 = batch2


                newx2, newedgeindex2, newedge2, newbatch2 = self.zijiegoumodel(newx2, newedgeindex2, newedge2, newbatch2)
                zijiegoutol_final2 = newx2.reshape(-1,self.num_query_token,self.args.gin_hidden_dim)

                # zijiegoutol_final2 = graph2pre
                graphembed2 = self.MLP(zijiegoutol_final2)

                zijiegou_final = (zijiegoutol_final1.unsqueeze(2)+zijiegoutol_final2.unsqueeze(1))/2
                zijiegou_final = zijiegou_final.reshape(-1,(self.num_query_token)*(self.num_query_token),self.args.gin_hidden_dim)
                attenscore = self.attenlinear(zijiegou_final)
                attenscore = attenscore.squeeze(-1)
                perm = torch.argsort(attenscore, dim=1, descending=True)
                top_indices = perm[:, :self.num_query_token]
                top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, zijiegou_final.size(-1))
                aggregated_features1 = torch.gather(zijiegou_final, dim=1, index=top_indices_expanded)

                mol_tokens = self.MLP(aggregated_features1)

                
                prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

                prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

                prompt_embeds[prompt_tokens.is_GraEmb1_token] = graphembed1.flatten(0, 1)
                prompt_embeds[prompt_tokens.is_GraEmb2_token] = graphembed2.flatten(0, 1)
                

                outputs = self.opt_model.generate(
                    # inputs_ids,
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
                # print('output_text',outputs.shape)
                #maxlen-1 * hidden_layers *(batchsize*numbeams*captions)*1*2048
                #分类
                if self.args.solve==False:
                    # pass
                    mlpdata  = outputs.hidden_states[26][-1][::num_beams].squeeze(1)
                    mlpclass = self.mlptoken(mlpdata)
                    mlp_probs = F.softmax(mlpclass, dim=1)
                    predicted_classes = torch.argmax(mlp_probs, dim=1)
                else:
                    # mlpdata1  = outputs.hidden_states[5][-1][::num_beams].squeeze(1)
                    # mlpdata2  = outputs.hidden_states[6][-1][::num_beams].squeeze(1)
                    # mlpdata3  = outputs.hidden_states[7][-1][::num_beams].squeeze(1)
                    # mlpdata4  = outputs.hidden_states[8][-1][::num_beams].squeeze(1)
                    # mlpdata5  = outputs.hidden_states[9][-1][::num_beams].squeeze(1)
                    # stacked_mlpdata = torch.stack([mlpdata1, mlpdata2, mlpdata3, mlpdata4, mlpdata5], dim=0)
                    # mlpdata = stacked_mlpdata.mean(dim=0)

                    mlpdata  = outputs.hidden_states[6][-1][::num_beams].squeeze(1)
                    predicted_classes = self.mlptoken_solve(mlpdata)
                if self.args.solve ==False:
                    zijiegoutoken1 = outputs.hidden_states[8][-1][::num_beams].squeeze(1)
                    zijiegoutoken1 = self.mlpzijiegou(zijiegoutoken1)
                    # predictions1 = torch.argmax(zijiegoutoken1, dim=1) 
                    sigmoid = nn.Sigmoid()
                    probabilities = sigmoid(zijiegoutoken1)
                    predictions1 = (probabilities > 0.5).float()

                    zijiegoutoken2 = outputs.hidden_states[19][-1][::num_beams].squeeze(1)
                    zijiegoutoken2 = self.mlpzijiegou(zijiegoutoken2)
                    # predictions2 = torch.argmax(zijiegoutoken2, dim=1) 
                    probabilities = sigmoid(zijiegoutoken2)
                    predictions2 = (probabilities > 0.5).float()

                output_text = self.opt_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                
                output_text = [text.strip() for text in output_text]
                # print(output_text)
                # return output_text
                if self.args.solve == False:
                    return output_text,predicted_classes,predictions1,predictions2
                else:
                    return output_text,predicted_classes
            
            else:
                graphs1 = samples['graphs1']
                zijiegou1 = samples['zijiegou1']
                prompt_tokens = samples['prompt_tokens']
                graphs2 = samples['graphs2']
                zijiegou2 = samples['zijiegou2']
                    
                zijiegou_embeds1= self.graph_encoder(graphs1,zijiegou1)
    

                mol_tokens1 = self.MLP(zijiegou_embeds1)
                
                zijiegou_embeds2= self.graph_encoder(graphs2,zijiegou2)
                mol_tokens2 = self.MLP(zijiegou_embeds2)

                mol_tokens = torch.cat([mol_tokens1,mol_tokens2], dim=1)

                
                prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
                # print(prompt_embeds.dtype,prompt_tokens.is_mol_token.dtype, mol_tokens.flatten(0, 1).dtype)
                prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

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
        else :
            graphs = samples['graphs']
            prompt_tokens = samples['prompt_tokens']
            # prompt_lens = samples['prompt_lens']
            # with self.maybe_autocast():
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks,
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state)
            
            prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)

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

