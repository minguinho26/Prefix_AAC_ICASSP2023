from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as nnf

# 평가를 위해 필요한 것들
from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple
from eval_metrics import evaluate_metrics
from terminaltables import AsciiTable
import pickle

device = 'cuda'

def Train(model, train_dataloader, test_dataloader, tokenizer,  epochs, model_name, beam_search) :
    
    model.train()
    model.to(device)
    
    warmup_steps = int((epochs * len(train_dataloader)) /6) # 총 weight update 횟수의 1/6은 warm-up 시기임
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader))
    
    min_loss = float("Inf")
    min_loss_file_path = ''
    
    epoch_eval_interval = int(epochs/3)
    
    for epoch in range(epochs) :
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
        avr_loss_per_epoch = 0.0
        for batch_i, (audio, tokens, mask, file_name) in enumerate(pbar) :
            
            audio = audio.to(device)
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            outputs = model(audio, tokens, mask)
            logits = outputs.logits[:, model.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]).to(device), tokens.flatten().to(device), ignore_index=0)
            avr_loss_per_epoch += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_description(f"Training Epoch {epoch}, Loss = {round(loss.item(), 5)}")
            
        if (epoch == epoch_eval_interval - 1) or (epoch == (2 * epoch_eval_interval) - 1) or (epoch == (epochs - 1)) :
            eval_model(model, test_dataloader, tokenizer, epoch, model_name, beam_search)
            model.train()

        avr_loss_per_epoch /= len(train_dataloader)
        
        param_file_path = "./params_" + model_name + "/Param_epoch_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), param_file_path)
        
        if min_loss > avr_loss_per_epoch :
            min_loss = avr_loss_per_epoch
            min_loss_file_path = param_file_path
    
    return min_loss_file_path

def eval_model(model, test_dataloader, tokenizer, epoch, model_name, beam_search) :
    model.eval()
    model.to(device)

    # 모아놨다가 한 번에 평가하자
    captions_pred: List[Dict] = []
    captions_gt: List[Dict] = []
    
    for i, (audio, tokens, mask, f_names) in enumerate(tqdm(test_dataloader, desc="Eval...")):
        with torch.no_grad():
            # 하나의 raw audio에 대해 5개의 caption이 등장
            
            # Test dataset은 audio, caption의 비율이 1:5다 
            # audio를 뽑고 5개당 하나씩 sampling하자
            audio = audio.to(device)
            
            index_list = []
            for i in range(int(audio.size()[0] / 5)) :
                index_list.append(5 * i)
            index_list = torch.tensor(index_list)
            
            audio = audio[index_list,:]
            
            if beam_search == True :
                generated_list = []
                text_list = model(audio, None, None, beam_search = beam_search)
                
                for j in range(len(text_list)):
                    generated_list.append(text_list[j][0])  
            else :
                generated_list = model(audio, None, None, beam_search = beam_search)
                
        
        for j in range(int(audio.size()[0] / 5)) :
            
            index_start_num = 5 * j
           
            caption_list = [tokenizer.decode(tokens[index_start_num + 0]).strip('!'),
                            tokenizer.decode(tokens[index_start_num + 1]).strip('!'),
                            tokenizer.decode(tokens[index_start_num + 2]).strip('!'),
                            tokenizer.decode(tokens[index_start_num + 3]).strip('!'),
                            tokenizer.decode(tokens[index_start_num + 4]).strip('!')]
            
            
            for k in range(len(caption_list)) :
                if caption_list[k][-1] != '.' :
                    caption_list[k] += '.'
            
            captions_pred.append({
                        'file_name': f_names[index_start_num], 
                        'caption_predicted': generated_list[j]})
            captions_gt.append({
                        'file_name': f_names[index_start_num],
                        'caption_1': caption_list[0],
                        'caption_2': caption_list[1],
                        'caption_3': caption_list[2],
                        'caption_4': caption_list[3],
                        'caption_5': caption_list[4]})
    
    # 전체 측정값을 한 번에 method에 넣어서 측정
    metrics = evaluate_metrics(captions_pred, captions_gt)
        
    #
    total_results = {}
    total_results['BLUE_1'] = metrics['bleu_1']['score']
    total_results['BLUE_2'] = metrics['bleu_2']['score']
    total_results['BLUE_3'] = metrics['bleu_3']['score']
    total_results['BLUE_4'] = metrics['bleu_4']['score']
    total_results['METEOR'] = metrics['meteor']['score']
    total_results['ROUGE_l'] = metrics['rouge_l']['score']
    total_results['CIDEr'] = metrics['cider']['score']
    total_results['SPICE'] = metrics['spice']['score']
    total_results['SPIDEr'] = metrics['spider']['score']  
    
    print("Pred, gt example")
    print("Pred :", generated_list[-1])
    print("Caption_1 :", caption_list[0])
    print("Caption_2 :", caption_list[1])
    print("Caption_3 :", caption_list[2])
    print("Caption_4 :", caption_list[3])
    print("Caption_5 :", caption_list[4])
    print()
    
    print("total result")
    print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["BLEU_1", format(round(float(total_results['BLUE_1']), 6), 'f')],
                        ["BLEU_2", format(round(float(total_results['BLUE_2']), 6), 'f')],
                        ["BLEU_3", format(round(float(total_results['BLUE_3']), 6), 'f')],
                        ["BLEU_4", format(round(float(total_results['BLUE_4']), 6), 'f')],
                        ["METEOR", format(round(float(total_results['METEOR']), 6), 'f')],
                        ["ROUGE_l", format(round(float(total_results['ROUGE_l']), 6), 'f')],
                        ["CIDEr", format(round(float(total_results['CIDEr']), 6), 'f')],
                        ["SPICE", format(round(float(total_results['SPICE']), 6), 'f')],
                        ["SPIDEr", format(round(float(total_results['SPIDEr']), 6), 'f')]
                    ]).table)    

    # 결과 저장 
    result_file_path = './eval_result/epoch_' + str(epoch) + '_' + model_name + '.pkl' 
    with open(result_file_path,'wb') as f:
        pickle.dump(total_results, f)