from pathlib import Path

from torch.utils.data import DataLoader

from data_handlers._clotho import ClothoDataset
from data_handlers.clotho_loader import _clotho_dataloader_preprocess


def get_clotho_dataloader(split, data_dir = Path('./data', 'data_splits')) :
    dataset = ClothoDataset(
        data_dir=data_dir,
        split=split,
        input_field_name='features',
        output_field_name='words_ind',
        load_into_memory=False)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=_clotho_dataloader_preprocess)
    
    return dataloader

def see_info(dataloader, split) : # split : 'development' or 'evaluation'
    print(split + "_dataloader's length :", len(dataloader))
    audio, token, f_name = next(iter(dataloader))
    
    print("size of", split + "_dataloader's audio batch :", audio.size())
    print("size of", split + "_dataloader's token batch :", token.size())