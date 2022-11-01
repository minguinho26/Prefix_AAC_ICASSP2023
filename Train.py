from transformers import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm
import time
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as nnf

# 평가를 위해 필요한 것들
from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple
from eval_metrics import evaluate_metrics
from terminaltables import AsciiTable
import pickle

def Train(model, LR, train_dataloader, test_dataloader, epochs, model_name, beam_search, device, Dataset = 'AudioCaps', test_dataloader_other_dataset = None) :
    
    model.train()
    model.to(device)
    
    warmup_steps = int((epochs * len(train_dataloader)) / 6)
    num_training_steps=epochs * len(train_dataloader)
    
    # AudioCaps를 사용할 경우 optimizer의 weight_decay는 0.01이 됨
    if Dataset == 'AudioCaps' :
#         optimizer = AdamW(model.parameters(), lr=LR, weight_decay = 0.01)
        optimizer = AdamW(
                          [
                            {"params": model.audio_encoder.parameters(), "lr": 2e-5},
                            {"params": model.temporal_mappingnetwork.parameters(), "lr": 5e-5},
                            {"params": model.global_mappingnetwork.parameters(), "lr": 5e-5},
                            {"params": model.language_header.parameters(), "lr": 2e-5},
                           ],lr=LR, weight_decay = 0.01)
    else :
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay = 0.02)
#         optimizer = AdamW(
#                           [
#                             {"params": model.audio_encoder.parameters(), "lr": 2e-5},
#                             {"params": model.temporal_mappingnetwork.parameters(), "lr": 5e-5},
#                             {"params": model.global_mappingnetwork.parameters(), "lr": 5e-5},
#                             {"params": model.language_header.parameters(), "lr": 2e-5},
#                            ],lr=LR, weight_decay = 0.02)
        
    scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    
    prefix_length = model.temporal_prefix_length + model.global_prefix_length
    
    if prefix_length == 0 :
        prefix_length = 26
    
    training_consumed_sec = 0
    
    for epoch in range(epochs) :
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
        total_loss_per_epopch = 0.0
        loss_add_count = 0.0
        
        train_start_time_per_epoch = time.time()
        
        for batch_i, (audio, tokens, mask, _) in enumerate(pbar) :
            
            audio = audio.to(device)
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            logits = model(audio, tokens, mask)[:, prefix_length - 1: -1]
                
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]).to(device), tokens.flatten().to(device), ignore_index=0)
                
            total_loss_per_epopch += loss.item()
            loss_add_count += 1.0
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            avr_loss = total_loss_per_epopch / loss_add_count
            pbar.set_description(f"Training Epoch {epoch}, Loss = {round(avr_loss, 5)}")
        
        training_consumed_sec += (time.time() - train_start_time_per_epoch)
        
        if (epoch >= 14) and ((epoch + 1) % 5 == 0) : 
            eval_model(model, test_dataloader, epoch, model_name, beam_search, device, Dataset, test_dataloader_other_dataset)
            model.train()
            
        if (epoch + 1 == 16) and (Dataset == 'AudioCaps' or Dataset == 'Fusion') :
#         if (epoch + 1 == 30) and (Dataset == 'AudioCaps' or Dataset == 'Fusion') :
            for param in model.audio_encoder.parameters():
                param.requires_grad = False
            
        # 모든 parameter를 튜닝 가능한 시점을 찾고 있다...
        elif Dataset == 'Clotho' and (epoch + 1 == 30) : # epoch : 60
            for param in model.audio_encoder.parameters():
                param.requires_grad = False
#             for param in model.language_header.parameters():
#                 param.requires_grad = True
            print("set encoder freeze!")
        
        param_file_path = "./Train_record/params_" + model_name + "/Param_epoch_" + str(epoch) + ".pt"
            
        torch.save(model.state_dict(), param_file_path)

    result_list = str(datetime.timedelta(seconds=training_consumed_sec)).split(".")
    print()
    print("Training time :", result_list[0])


def eval_model(model, test_dataloader, epoch, model_name, beam_search, device, Dataset, test_dataloader_other_dataset = None) :
    
    model.eval()
    model.to(device)

    # 모아놨다가 한 번에 평가하자

    captions_pred: List[Dict] = []
    captions_gt: List[Dict] = []
        
    captions_pred_other_dataset: List[Dict] = []
    captions_gt_other_dataset: List[Dict] = []
    
    for i, (audio, captions, f_names) in enumerate(tqdm(test_dataloader, desc="Eval using dataset...")):
        with torch.no_grad():
            # 하나의 raw audio에 대해 5개의 caption이 등장
            
            # Test dataset은 audio, caption의 비율이 1:5다 
            # Batch size를 5로 설정했음. 0번 인덱스 값만 사용할거임
            audio = audio.to(device)
            
            audio = audio[0,:].unsqueeze(0)
            
            if beam_search == True :
                pred_caption = model(audio, None, beam_search = True)[0][0]
            else :
                pred_caption = model(audio, None, beam_search = False)[0]

        captions_pred.append({
                            'file_name': f_names[0], 
                            'caption_predicted': pred_caption})
        captions_gt.append({
                            'file_name': f_names[0],
                            'caption_reference_01': captions[0],
                            'caption_reference_02': captions[1],
                            'caption_reference_03': captions[2],
                            'caption_reference_04': captions[3],
                            'caption_reference_05': captions[4]})
       
    # 전체 측정값을 한 번에 method에 넣어서 측정
    metrics = evaluate_metrics(captions_pred, captions_gt)
    
    if test_dataloader_other_dataset == None :
        return [metrics, captions_pred, captions_gt]
    else :
        
        print("==========================================================================================")
        
        for i, (audio, captions, f_names) in enumerate(tqdm(test_dataloader_other_dataset, desc="Eval using other dataset...")):
            with torch.no_grad():
                # 하나의 raw audio에 대해 5개의 caption이 등장

                # Test dataset은 audio, caption의 비율이 1:5다 
                # Batch size를 5로 설정했음. 0번 인덱스 값만 사용할거임
                audio = audio.to(device)

                audio = audio[0,:].unsqueeze(0)

                if beam_search == True :
                    pred_caption = model(audio, None, beam_search = True)[0][0]
                else :
                    pred_caption = model(audio, None, beam_search = False)[0]

            captions_pred_other_dataset.append({
                                'file_name': f_names[0], 
                                'caption_predicted': pred_caption})
            captions_gt_other_dataset.append({
                                'file_name': f_names[0],
                                'caption_reference_01': captions[0],
                                'caption_reference_02': captions[1],
                                'caption_reference_03': captions[2],
                                'caption_reference_04': captions[3],
                                'caption_reference_05': captions[4]})

        # 전체 측정값을 한 번에 method에 넣어서 측정
        metrics_other_dataset = evaluate_metrics(captions_pred_other_dataset, captions_gt_other_dataset)


        return [metrics, captions_pred, captions_gt], [metrics_other_dataset, captions_pred_other_dataset, captions_gt_other_dataset]