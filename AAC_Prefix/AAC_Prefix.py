import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
import math
import copy

from torch.nn import functional as nnf

from util import *
from AAC_Prefix.PANNs.CNN14 import Cnn14 # audio encoder : PANNs
from .Transformer import * # transformer

num_head = 8

# PE from PyTorch(link : ) 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MappingNetwork_forTemporalFeature(nn.Module):
    def forward(self, x):
        
        x = self.conv(x) # [batch_size, 2048, 15, 2] -> [batch_size, 768, 15, 1]
        x = self.bn_conv(x) 
        
        if self.Dataset == 'AudioCaps':
            x = self.relu_conv(x)
        
        x = torch.squeeze(x, 3) # [batch_size, 768, 15, 2] -> [batch_size, 768, 15]
        
        x = x.permute(2, 0, 1).contiguous() # [batch_size, 768, 15] -> [15, batch_size, 768]
        x = self.pos_encoder(x) # positional encoding
        
        x = x.permute(1, 0, 2).contiguous() # [15, batch_size, 768] -> [batch_size, 15, 768]
        
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, -self.prefix_length:]
        return out

    def __init__(self, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, device = 'cuda:1', Dataset = 'AudioCaps'):
        super(MappingNetwork_forTemporalFeature, self).__init__()
        
        self.Dataset = Dataset
        
        self.prefix_length = prefix_length

        self.device = device
        self.transformer = Transformer(dim_embedding, num_head, num_layers)
        
        # 시간대역 별로 특성을 분석
        self.conv = nn.Conv2d(2048, dim_embedding, (1, 2), stride=(1, 1), padding=(0, 0)) # [2048, 15, 2] -> [768, 15, 1]
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        
        self.bn_conv = nn.BatchNorm2d(dim_embedding)
        self.relu_conv = nn.ReLU()
        
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.prefix_const)
        
        self.pos_encoder = PositionalEncoding(d_model=dim_embedding, dropout = 0.5) # positional encoding
        
        print("temporal feature ver's mapping network : num_head =", num_head, "num_layers =", num_layers, "prefix_vector_lentgh =", prefix_length)
        

class MappingNetwork_forGlobalFeature(nn.Module):

    def forward(self, x):
        dummy_val = torch.zeros(x.size()[0], 1).to(self.device)
        x = torch.cat((x, dummy_val), dim=1) # [batch_size, 527] -> [batch_size, 528]
        
        x = (x.unsqueeze(1)).unsqueeze(1) # [batch_size, 528] -> [batch_size, 1, 1, 528]
        x = self.conv(x) # [batch_size, 1, 1, 528] -> [batch_size, 768, 1, 11] 
        x = self.bn_conv(x)
        
        if self.Dataset == 'AudioCaps' :
            x = self.relu_conv(x)
        
        x = torch.squeeze(x, 2) # [batch_size, 768, 1, 11] -> [batch_size, 768, 11]
        
        x = x.permute(0, 2, 1).contiguous() # [batch_size, 768, 11] -> [batch_size, 11, 768]
        
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, -self.prefix_length:]
        return out

    def __init__(self, dim_embedding: int, prefix_length: int, clip_length: int, num_layers : int = 8, device = 'cuda:1', Dataset = 'AudioCaps'):
        super(MappingNetwork_forGlobalFeature, self).__init__()

        self.device = device
        self.Dataset = Dataset

        self.prefix_length = prefix_length
        self.transformer = Transformer(dim_embedding, num_head, num_layers)
        
        self.conv = nn.Conv2d(1, 768, (1, 48), stride=(1, 48), padding=(0, 0))
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        
        self.bn_conv = nn.BatchNorm2d(dim_embedding)
        self.relu_conv = nn.ReLU()
        
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.prefix_const)
        
        print("global feature ver's mapping network : num_head =", num_head, "num_layers =", num_layers, "prefix_vector_lentgh =", prefix_length)


class AAC_Prefix(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, (self.temporal_prefix_length + self.global_prefix_length), dtype=torch.int64, device=device)
    
    def get_logits_for_inference(self, generated) :
        
        out = self.gpt(inputs_embeds=generated)
        out_hidden_states = out[0]
        logits = self.language_header(out_hidden_states)

        # The first word in own vocabulary is '!'. It is not used for creating sentence, just for padding.
        # Therefore we set the value about this word to 0.
        if self.vocab_size != None :
            logits[:,:,0] = 0.0 
        
#         if self.Dataset == 'Clotho' and self.vocab_size != None :
#             logits_for_clotho = self.language_header_only_for_clotho(out_hidden_states)
#             logits = torch.cat((logits, logits_for_clotho), dim=2)
        
        return logits
    
    def generate_beam(self, prefix_projections, beam_size = 5) :
        
        entry_count = prefix_projections.size()[0]
        entry_length = 67
        temperature=1.0
                
        if self.vocab_size == None :
            stop_token_index = self.tokenizer.encode(".")[0]
        else :
            stop_token_index = 13
        
        output_texts_list = []
        
        for entry_idx in range(entry_count):
            
            generated = prefix_projections[entry_idx,:,:].unsqueeze(0)
            scores = None
            tokens = None
            seq_lengths = torch.ones(beam_size, device=self.device)
            is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)
            
            for i in range(entry_length):
                
                logits = self.get_logits_for_inference(generated)
                
                logits = logits[:, -1, :] / (temperature)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                        beam_size, -1
                    )
                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
#                     next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.gpt.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                del logits 
                if is_stopped.all():
                    del generated
                    break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.decode(output[: int(length)])
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]

            output_texts_list.append(output_texts)
        
        return output_texts_list

    def generate(self, prefix_projections) :
        temperature = 1.0
        entry_length = 67
        top_p = 0.8
        
        if self.vocab_size == None :
            stop_token_index = self.tokenizer.encode(".")[0]
        else :
            stop_token_index = 13
            
        filter_value = -float("Inf")
        generated_list = []
        
        entry_count = prefix_projections.size()[0]
        
        
        for entry_idx in range(entry_count):

            generated = prefix_projections[entry_idx]
            
            tokens = None 
            
            for i in range(entry_length):
                
                logits = self.get_logits_for_inference(generated)

                logits = logits[:, -1, :] / (temperature)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                            nnf.softmax(sorted_logits, dim=-1), dim=-1
                        )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = self.gpt.wte(next_token)
                
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.tokenizer.decode(output_list)
            generated_list.append(output_text)
        return generated_list 

    def forward(self, audio, tokens = None, mask = None, labels = None, beam_search = False):
        
        temporal_feature, global_feature = self.audio_encoder(audio)
        
        if self.temporal_prefix_length > 0 :
            temporal_prefix_vector = self.temporal_mappingnetwork(temporal_feature).view(-1, self.temporal_prefix_length, self.gpt_embedding_size)
        elif self.global_prefix_length + self.temporal_prefix_length == 0  :
            temporal_feature = temporal_feature.permute(0,2,1,3).contiguous()
            temporal_feature = torch.reshape(temporal_feature, (temporal_feature.size()[0], temporal_feature.size()[1], -1))  
            temporal_prefix_vector = self.temporal_mappingnetwork(temporal_feature)
            
        if self.global_prefix_length > 0 :
            global_prefix_vector = self.global_mappingnetwork(global_feature).view(-1, self.global_prefix_length, self.gpt_embedding_size)
        elif self.global_prefix_length + self.temporal_prefix_length == 0 :
            global_prefix_vector = self.global_mappingnetwork(global_feature)
            global_prefix_vector = global_prefix_vector.view(global_feature.size()[0], 11, 768)

        if self.temporal_prefix_length > 0 and self.global_prefix_length == 0 :
            prefix_vectors = temporal_prefix_vector
        elif self.temporal_prefix_length == 0 and self.global_prefix_length > 0 :
            prefix_vectors = global_prefix_vector
        else :
            prefix_vectors = torch.cat((temporal_prefix_vector, global_prefix_vector), dim=1) 
            
            
#         prefix_vectors = torch.cat((temporal_prefix_vector, global_prefix_vector), dim=1) 
#         prefix_vectors = temporal_prefix_vector
#         prefix_vectors = global_prefix_vector
        if self.training :
            embedding_text = self.gpt.wte(tokens.to(self.device))
            embedding_cat = torch.cat((prefix_vectors, embedding_text), dim=1)
            
            out = self.gpt(inputs_embeds=embedding_cat.to(self.device), attention_mask=mask.to(self.device))
            out_hidden_states = out[0]
            
            logits = self.language_header(out_hidden_states)
            
            if self.vocab_size != None :
                logits[:,:,0] = 0.0 # '!' token은 사용하지 않기 때문에 예측하지 않게끔 만든다
            
#             if self.Dataset == 'Clotho' and self.vocab_size != None :
#                 logits_for_clotho = self.language_header_only_for_clotho(out_hidden_states)
#                 logits = torch.cat((logits, logits_for_clotho), dim=2)
                
            return logits
        else :
            if beam_search == True :
                return self.generate_beam(prefix_vectors)
            else :   
                return self.generate(prefix_vectors)
            

    def __init__(self, audio_encoder, tokenizer,
                 encoder_freeze = True, decoder_freeze = True, 
                 vocab_size = None, vocab_size_only_clotho = None, Dataset = 'AudioCaps', 
                 prefix_size_dict = {"temporal_prefix_size" : 10, "global_prefix_size" : 10}, 
                 temporal_num_layers = 2, global_num_layers = 2,
                 pretrain_fromAudioCaps = False, device = 'cuda:1'):
        
        super(AAC_Prefix, self).__init__()
        self.device = device
        self.Dataset = Dataset
        self.vocab_size = vocab_size
        self.SAMPLE_RATE = 16000
        
        self.temporal_prefix_length = prefix_size_dict["temporal_prefix_size"]
        self.global_prefix_length = prefix_size_dict["global_prefix_size"]
        
        # same with prefix_length
        temporal_clip_length = prefix_size_dict["temporal_prefix_size"]
        global_clip_length = prefix_size_dict["global_prefix_size"]
        
        self.tokenizer = tokenizer
        self.audio_encoder = audio_encoder
        self.gpt = GPT2Model.from_pretrained("gpt2")

        self.gpt_embedding_size = self.gpt.wte.weight.shape[1] # 768
        
        
        if self.temporal_prefix_length > 0 :

            self.temporal_mappingnetwork = MappingNetwork_forTemporalFeature(dim_embedding = self.gpt_embedding_size, 
                                        prefix_length = self.temporal_prefix_length, clip_length = temporal_clip_length, 
                                        num_layers = temporal_num_layers, device = device, Dataset = Dataset)   
        else :
            print("no temporal mapping network!")
            self.temporal_mappingnetwork = nn.Linear(2048*2, 768, bias = False)
            nn.init.kaiming_uniform_(self.temporal_mappingnetwork.weight)  
            
        if self.global_prefix_length > 0 :
            self.global_mappingnetwork = MappingNetwork_forGlobalFeature(dim_embedding = self.gpt_embedding_size, 
                                                prefix_length = self.global_prefix_length, clip_length = global_clip_length, 
                                                num_layers = global_num_layers, device = device, Dataset = Dataset)
        else :
            print("no global mapping network!")
            self.global_mappingnetwork = nn.Linear(527, 11*768, bias = False)
            nn.init.kaiming_uniform_(self.global_mappingnetwork.weight)  
        
        self.language_header = None
        
        if vocab_size == None : # If we do not use own vocaburaly
            self.language_header = nn.Linear(768, 50257, bias=False) # 50257 : original vocabulary size of GPT2
            header_gpt2_header_params = './AAC_Prefix/PreTrained_GPT2Header.pt'
            self.language_header.load_state_dict(torch.load(header_gpt2_header_params)) # use pre-trained header
        else :
            self.language_header = nn.Linear(768, vocab_size, bias=False)
            nn.init.kaiming_uniform_(self.language_header.weight)    
            print("use custom header!")
            
        if pretrain_fromAudioCaps == True :
            
            print("Get Pre-traiend Params")
            
            if vocab_size != None : 
                folder_name = 'Custom'
            elif vocab_size == None : # GPT2 Tokenizer
                folder_name = 'GPT2'
                  
            temporal_mappingnetwork_pt_name = 'temporal_mappingnetwork_' + str(folder_name) + '_in_Audiocaps.pt'
            global_mappingnetwork_pt_name = 'global_mappingnetwork_' + str(folder_name) + '_in_Audiocaps.pt'
            language_header_pt_name = 'language_header_' + str(folder_name) + '_in_Audiocaps.pt'

            temporal_mappingnetwork_path = './AAC_Prefix/pre_trained_params_from_audiocaps/' + \
                                       str(folder_name) + '/' + temporal_mappingnetwork_pt_name
            global_mappingnetwork_path = './AAC_Prefix/pre_trained_params_from_audiocaps/' + \
                                         str(folder_name) + '/' + global_mappingnetwork_pt_name
            language_header_path = './AAC_Prefix/pre_trained_params_from_audiocaps/' + \
                                         str(folder_name) + '/' + language_header_pt_name
            
            if self.temporal_prefix_length == 15 :
                    self.temporal_mappingnetwork.load_state_dict(torch.load(temporal_mappingnetwork_path))
            if self.global_prefix_length == 11 :
                self.global_mappingnetwork.load_state_dict(torch.load(global_mappingnetwork_path))
            
            if folder_name != 'GPT2' :
                print("Get Pre-traiend language header")
                
                temp_header_1 = nn.Linear(768, 7911, bias = False)
                temp_header_1.load_state_dict(torch.load(language_header_path))
                
                temp_header_2 = nn.Linear(768, vocab_size - 7911, bias = False)
                nn.init.kaiming_uniform_(temp_header_2.weight)
                
                with torch.no_grad():
                    self.language_header.weight[:7911,:] = temp_header_1.weight
                    self.language_header.weight[7911:,:] = temp_header_2.weight
                

        if encoder_freeze == True :
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            print("Encoder freezing")
            
        if decoder_freeze == True :
            for param in self.gpt.parameters():
                param.requires_grad = False
            is_header_freeze = False    
#             for param in self.language_header.parameters():
#                 param.requires_grad = False
#                 is_header_freeze = True
            print("GPT2 freezing")
    
            if is_header_freeze == True :
                print("header freezing")
            else :
                print("header trainable!")
            
                
def get_AAC_Prefix(tokenizer, 
                    vocab_size = None, Dataset = 'AudioCaps',
                    prefix_size_dict = {"temporal_prefix_size" : 10, "global_prefix_size" : 10}, 
                    transformer_num_layers = None, encoder_freeze = True, decoder_freeze = True, 
                    pretrain_fromAudioCaps = False, device = 'cuda:1') :
    
    # PANNS
    audio_encoder = Cnn14(sample_rate=16000, window_size=512, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=527)
    
    folder_name = None

    if vocab_size != None :
        print("use Custom Tokenizer")
        if Dataset == 'Clotho' : 
            folder_name = 'Custom' 
    else :
        print("use GPT2 Tokenizer")
        folder_name = 'GPT2'
    
    vocab_size_only_clotho = None # the number of words only in Clotho
    if vocab_size != None and folder_name != None :
        vocab_size_only_clotho = vocab_size - 7911
    
    if pretrain_fromAudioCaps == False :
        checkpoint_path = "./AAC_Prefix/PANNs/Cnn14_16k_mAP=0.438.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        audio_encoder.load_state_dict(checkpoint['model'])
    else :
        audio_encoder_pt_name = 'audio_encoder_' + str(folder_name) + '_in_Audiocaps.pt'
        audio_encoder_path = './AAC_Prefix/pre_trained_params_from_audiocaps/'  + \
                              str(folder_name) + '/' + audio_encoder_pt_name
           
        audio_encoder.load_state_dict(torch.load(audio_encoder_path))
    
    audio_encoder = audio_encoder.to(device)
    
    temporal_num_layers = transformer_num_layers["temporal_num_layers"]
    global_num_layers = transformer_num_layers["global_num_layers"]

    model = AAC_Prefix(audio_encoder, tokenizer,
                        encoder_freeze, decoder_freeze, 
                        vocab_size, vocab_size_only_clotho, Dataset,
                        prefix_size_dict = prefix_size_dict, 
                        temporal_num_layers = temporal_num_layers, global_num_layers = global_num_layers, 
                        pretrain_fromAudioCaps = pretrain_fromAudioCaps, device = device)
    
    return model.to(device)


def get_model_in_table(table_num, setting_num, device) :
    transformer_num_layers = {"temporal_num_layers" : 4, "global_num_layers" : 4}
    prefix_size_dict = {"temporal_prefix_size" : 15, "global_prefix_size" : 11}
    
    if (table_num == 1 and setting_num == 1) or (table_num == 2 and setting_num == 2) :
        Dataset = 'Clotho' 
    elif (table_num == 1 and setting_num == 2) or (table_num == 1 and setting_num == 3) or (table_num == 2 and setting_num == 1) or (table_num == 2 and setting_num == 3) :
        Dataset = 'AudioCaps'
    
    if setting_num == 1 :
        tokenizer_type = 'Custom'
        tokenizer = tokenizer_forCustomVocab(Dataset = Dataset)
        vocab_size = len(tokenizer.vocab)
    else :
        tokenizer_type = 'GPT2'
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        vocab_size = None
    
    model = get_AAC_Prefix(tokenizer, 
                        vocab_size = vocab_size, Dataset = Dataset,
                        prefix_size_dict = prefix_size_dict, transformer_num_layers = transformer_num_layers, 
                        encoder_freeze = True, decoder_freeze = True,
                        pretrain_fromAudioCaps = False, device = device)
    
    if setting_num != 3 :
        model_path = 'Params_in_Table/Table' + str(table_num) + '_' + str(setting_num) + '_params.pt'
    else :
        model_path = 'Params_in_Table/Params_Overall_Dataset.pt'
    
    params = torch.load(model_path, map_location = device)
    
    model.load_state_dict(params) 
    
    return model