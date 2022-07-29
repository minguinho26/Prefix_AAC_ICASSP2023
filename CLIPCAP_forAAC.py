import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

from torch.nn import functional as nnf
from typing import Optional, Tuple

from audio_encoder import get_audio_encoder # audio encoder

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
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
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    
    def generate_beam(self, prefix_projections, tokens, beam_size = 5) :
        
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

                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits
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
                next_token_embed = self.gpt.transformer.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                del outputs, logits 
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

        
    
    def generate(self, prefix_projections, tokens) :
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
                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits
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
                next_token_embed = self.gpt.transformer.wte(next_token)
                
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
        
        prefix = self.audio_encoder(audio)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        if self.training :
            embedding_text = self.gpt.transformer.wte(tokens.to('cuda'))
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            out = self.gpt(inputs_embeds=embedding_cat.to('cuda'), labels=labels, attention_mask=mask.to('cuda'))
            return out
        else :
            if beam_search == True :
                return self.generate_beam(prefix_projections, tokens)
            else :   
                return self.generate(prefix_projections, tokens)
            

    def __init__(self, audio_encoder, tokenizer, encoder_freeze = True, decoder_freeze = True, prefix_length = 10, clip_length = 10, prefix_size = 1024,
                 num_layers = 4, mapping_type = 'MLP'): # Transformer의 Layer 수를 4개로 줄여봤음(22.7.12)
        super(ClipCap_AAC, self).__init__()
        self.device = 'cuda'
        self.prefix_length = prefix_length
        self.tokenizer = tokenizer
        self.audio_encoder = audio_encoder
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == 'MLP':
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        elif mapping_type == 'TRANSFORMER':
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
        
        if encoder_freeze == True :
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        else :
            # fbsp만 Trainable하게 만든다. 나머지 부분(Feature Extraction 등은 Freezing)
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.fbsp.parameters():
                param.requires_grad = True
        
        if decoder_freeze == True :
            for param in self.gpt.parameters():
                param.requires_grad = False


def get_ClipCap_AAC(tokenizer, mapping_type = 'MLP', encoder_freeze = True, decoder_freeze = True) :
    
    audio_encoder = get_audio_encoder()

    model = ClipCap_AAC(audio_encoder, tokenizer, encoder_freeze, decoder_freeze, mapping_type = mapping_type)
    
    return model

# contrastive loss를 통해 clotho caption에 적응한 audio encoder를 사용
def get_ClipCap_AAC_adopt_contrastive_loss(audioenc_param_path, tokenizer, mapping_type = 'MLP', encoder_freeze = True, decoder_freeze = True) :
    audio_encoder = get_audio_encoder()
    audio_encoder.load_state_dict(torch.load(audioenc_param_path, map_location='cuda:0'))

    model = ClipCap_AAC(audio_encoder, tokenizer, encoder_freeze, decoder_freeze, mapping_type = mapping_type)
    
    return model



