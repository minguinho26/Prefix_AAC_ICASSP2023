import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Model
import numpy as np
import math

from torch.nn import functional as nnf
from typing import Optional, Tuple

from .audio_encoder import get_audio_encoder # audio encoder
from ClipCap_forAAC.PANNs.CNN14 import Cnn14 # audio encoder : PANNs
from .Transformer import * # transformer

# Pytorch에서 제공해주는 PE 
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

# PANNs를 Audio Encoder로 사용했을 때 Mapping Network들임
# 주석으로 달린 수치는 Clotho dataset을 사용한다 가정했을 때 측정되는 size들임
class TransformerMapper_forAudioFeature(nn.Module):

    def forward(self, x):
        
        x = self.conv(x) # [batch_size, 2048, 15, 2] -> [batch_size, 768, 15, 1]
        x = self.bn_conv(x) 
        
        x = torch.squeeze(x, 3) # [batch_size, 768, 15, 2] -> [batch_size, 768, 15]
        
        x = x.permute(2, 0, 1) # [batch_size, 768, 15] -> [15, batch_size, 768]
        x = self.pos_encoder(x) # positional encoding
        
        x = x.permute(1, 0, 2) # [15, batch_size, 768] -> [batch_size, 15, 768]
        
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, device = 'cuda'):
        super(TransformerMapper_forAudioFeature, self).__init__()
        self.clip_length = clip_length
        
        self.device = device
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        # 시간대역 별로 특성을 분석
        self.conv = nn.Conv2d(2048, dim_embedding, (1, 2), stride=(1, 1), padding=(0, 0)) # [2048, 15, 2] -> [768, 15, 1]
        self.bn_conv = nn.BatchNorm2d(dim_embedding)
        
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        
        self.pos_encoder = PositionalEncoding(d_model=dim_embedding, dropout = 0.5) # positional encoding
        
        
# 527-d vector를 받음
class TransformerMapper_forSemanticFeature(nn.Module):

    def forward(self, x):
        
        # 받은 vector는 527 종류의 오디오가 각각 얼마나 있는지 나타낸 값이다
        # 이 vector 뒤에 0이 들어있는 값을 붙이면 528 = 11*48 차원의 vector가 된다. 여기서 11종류의 정보를 뽑아내보자.
        
        dummy_val = torch.zeros(x.size()[0], 1).to(self.device)
        x = torch.cat((x, dummy_val), dim=1) # [batch_size, 527] -> [batch_size, 528]
        
        x = (x.unsqueeze(1)).unsqueeze(1) # [batch_size, 528] -> [batch_size, 1, 1, 528]
        x = self.conv(x) # [batch_size, 1, 1, 528] -> [batch_size, 768, 1, 11] 
        x = self.bn_conv(x)
        x = torch.squeeze(x, 2) # [batch_size, 768, 1, 11] -> [batch_size, 768, 11]
        x = x.permute(0, 2, 1) # [batch_size, 768, 11] -> [batch_size, 11, 768]
        
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, device = 'cuda'):
        super(TransformerMapper_forSemanticFeature, self).__init__()

        self.device = device

        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        
        self.conv = nn.Conv2d(1, 768, (1, 48), stride=(1, 48), padding=(0, 0))
        self.bn_conv = nn.BatchNorm2d(dim_embedding)
        
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        
class MLP(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        
class ClipCap_AAC(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, (self.audio_prefix_length + self.semantic_prefix_length), dtype=torch.int64, device=device)
    
    def generate_beam(self, prefix_projections, beam_size = 5) :
        
        entry_count = prefix_projections.size()[0]
        entry_length = 67
        temperature=1.0
        stop_token_index = self.tokenizer.encode(".")[0]
        
        output_texts_list = []
        
        for entry_idx in range(entry_count):
            
            generated = prefix_projections[entry_idx,:,:].unsqueeze(0)
            scores = None
            tokens = None
            seq_lengths = torch.ones(beam_size, device=self.device)
            is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)
            
            for i in range(entry_length):
                out = self.gpt(inputs_embeds=generated)
                out_hidden_states = out[0]
                logits = self.language_header(out_hidden_states)
                
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
        stop_token_index = self.tokenizer.encode(".")[0]
        filter_value = -float("Inf")
        generated_list = []
        
        entry_count = prefix_projections.size()[0]
        
        
        for entry_idx in range(entry_count):

            generated = prefix_projections[entry_idx]
            
            tokens = None # caption만들어줄 때마다 초기화 필수
            
            for i in range(entry_length):
                out = self.gpt(inputs_embeds=generated)
                out_hidden_states = out[0]
                logits = self.language_header(out_hidden_states)

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
        
        audio_feature, semantic_feature = self.audio_encoder(audio)
        
        audio_prefix_projections = self.audio_clip_project(audio_feature).view(-1, self.audio_prefix_length, self.gpt_embedding_size)
        
        semantic_prefix_projections = self.semantic_clip_project(semantic_feature).view(-1, self.semantic_prefix_length, self.gpt_embedding_size)
        
                
        prefix_projections = torch.cat((audio_prefix_projections, semantic_prefix_projections), dim=1) # 기존 제안
        if self.training :
            embedding_text = self.gpt.wte(tokens.to(self.device))
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            
            out = self.gpt(inputs_embeds=embedding_cat.to(self.device), attention_mask=mask.to(self.device))
            out_hidden_states = out[0]
            
            logits = self.language_header(out_hidden_states)
            
            return semantic_feature, logits
        else :
            if beam_search == True :
                return self.generate_beam(prefix_projections)
            else :   
                return self.generate(prefix_projections)
            

    def __init__(self, audio_encoder, tokenizer, encoder_freeze = True, decoder_freeze = True, 
                 vocab_size = None,
                 prefix_size_dict = {"audio_prefix_size" : 10, "semantic_prefix_size" : 10}, 
                 audio_prefix_size = 2048, semantic_prefix_size = 1024, 
                 audio_num_layers = 2, semantic_num_layers = 2,
                 mapping_type = 'MLP', pretrain_fromAudioCaps = False, device = 'cuda'):
        
        super(ClipCap_AAC, self).__init__()
        self.device = device
        self.audio_prefix_length = prefix_size_dict["audio_prefix_size"]
        self.semantic_prefix_length = prefix_size_dict["semantic_prefix_size"]
        
        # prefix와 크기 통일
        audio_clip_length = prefix_size_dict["audio_prefix_size"]
        semantic_clip_length = prefix_size_dict["semantic_prefix_size"]
        
        self.tokenizer = tokenizer
        self.audio_encoder = audio_encoder
        self.gpt = GPT2Model.from_pretrained("gpt2")
        
        if vocab_size == None :
            self.language_header = nn.Linear(768, 50257, bias=False) # 50257 : GPT2에서 쓰던 vocab의 사이즈
        else :
            self.language_header = nn.Linear(768, vocab_size, bias=False) # vocab_size : custom vocabulary의 사이즈

        self.gpt_embedding_size = self.gpt.wte.weight.shape[1] # 768
        
        if mapping_type == 'MLP':
            self.audio_clip_project = MLP((audio_prefix_size, (self.gpt_embedding_size * self.audio_prefix_length) // 2,
                                     self.gpt_embedding_size * self.audio_prefix_length))
            
            self.semantic_clip_project = MLP((semantic_prefix_size, (self.gpt_embedding_size * self.semantic_prefix_length) // 2,
                                     self.gpt_embedding_size * self.semantic_prefix_length))
            
        elif mapping_type == 'TRANSFORMER':
            self.audio_clip_project = TransformerMapper_forAudioFeature(audio_prefix_size, self.gpt_embedding_size, 
                                                        self.audio_prefix_length, audio_clip_length, audio_num_layers, device)
            
            self.semantic_clip_project = TransformerMapper_forSemanticFeature(semantic_prefix_size, self.gpt_embedding_size, 
                                                           self.semantic_prefix_length, semantic_clip_length, semantic_num_layers, device)
            
        if encoder_freeze == True :
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        if decoder_freeze == True :
            for param in self.gpt.parameters():
                param.requires_grad = False
        
        if pretrain_fromAudioCaps == True :
            checkpoint_path = "./ClipCap_forAAC/pre_trained_params_from_audiocaps/audio_clip_project_SOTA_in_Audiocaps.pt"
            self.audio_clip_project.load_state_dict(torch.load(checkpoint_path))
            
            checkpoint_path = "./ClipCap_forAAC/pre_trained_params_from_audiocaps/semantic_clip_project_SOTA_in_Audiocaps.pt"
            self.semantic_clip_project.load_state_dict(torch.load(checkpoint_path))
            
            if vocab_size == None : # GPT2 tokenizer를 사용할 경우, huggingface에서 제공하는 header를 사용
                checkpoint_path = './ClipCap_forAAC/pre_trained_params_from_audiocaps/PreTrained_Header_fromAudioCaps.pt'
                self.language_header.load_state_dict(torch.load(checkpoint_path))
        else :
            if vocab_size == None : # GPT2 tokenizer를 사용할 경우, huggingface에서 제공하는 header를 사용
                header_gpt2_header_params = './ClipCap_forAAC/PreTrained_GPT2Header.pt'
                self.language_header.load_state_dict(torch.load(header_gpt2_header_params)) # Huggingface에서 사전학습된 header
            
                
def get_ClipCap_AAC(tokenizer, vocab_size = None, mapping_type = 'MLP', Dataset = 'AudioCaps',
                    prefix_size_dict = {"audio_prefix_size" : 10, "semantic_prefix_size" : 10}, 
                    transformer_num_layers = None, encoder_freeze = True, decoder_freeze = True, 
                    pretrain_fromAudioCaps = False, device = 'cuda') :
    
    # audio_encoder = get_audio_encoder()
    # PANNS
    audio_encoder = Cnn14(sample_rate=16000, window_size=512, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=527)
    
    if pretrain_fromAudioCaps == False :
        checkpoint_path = "./ClipCap_forAAC/PANNs/Cnn14_16k_mAP=0.438.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        audio_encoder.load_state_dict(checkpoint['model'])
    else :
        checkpoint_path = "./ClipCap_forAAC/audio_encoder_SOTA_in_Audiocaps.pt"
        audio_encoder.load_state_dict(torch.load(checkpoint_path))
    
    # Clotho는 30초짜리 audio를 쓰는데 ours는 10초짜리 오디오만 처리함. 그래서 30초짜리 audio로 뽑은 값을 10초짜리 오디오로 뽑은 값과 같이 압축시켜주는 module이 필요함
    # 그러한 module을 추가해주는 method가 add_compress_feature()임
    if Dataset == 'Clotho' :
        audio_encoder.add_compress_feature()
    
    audio_encoder = audio_encoder.to(device)
    
    audio_num_layers = transformer_num_layers["audio_num_layers"]
    semantic_num_layers = transformer_num_layers["semantic_num_layers"]

    model = ClipCap_AAC(audio_encoder, tokenizer, 
                        encoder_freeze, decoder_freeze, 
                        vocab_size, 
                        prefix_size_dict = prefix_size_dict, 
                        audio_num_layers = audio_num_layers, semantic_num_layers = semantic_num_layers, 
                        mapping_type = mapping_type, pretrain_fromAudioCaps = pretrain_fromAudioCaps, device = device)
    
    return model.to(device)