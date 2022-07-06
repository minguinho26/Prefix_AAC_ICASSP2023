#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/minkyukim/models/CLIP_AAC_implementation_project/ACLIP')

from typing import List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from numpy import load as np_load, ndarray

import numpy as np
import librosa
import pandas as pd

SAMPLE_RATE = 44100

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


class ClothoDataset(Dataset):

    def __init__(self,
                 data_dir: Path,
                 split: str,
                 input_field_name: str,
                 output_field_name: str,
                 load_into_memory: bool
                 ) \
            -> None:
        """Initialization of a Clotho dataset object.

        :param data_dir: Data directory with Clotho dataset files.
        :type data_dir: pathlib.Path
        :param split: The split to use (`development`, `validation`)
        :type split: str
        :param input_field_name: Field name for the input values
        :type input_field_name: str
        :param output_field_name: Field name for the output (target) values.
        :type output_field_name: str
        :param load_into_memory: Load the dataset into memory?
        :type load_into_memory: bool
        """
        super(ClothoDataset, self).__init__()
        the_dir: Path = data_dir.joinpath(split)

        self.examples: List[Path] = sorted(the_dir.iterdir())
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory
        self.split = split
        
        if load_into_memory:
            self.examples: List[ndarray] = [
                np_load(str(f), allow_pickle=True)
                for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)
    
    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray, Path]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values, and the Path of the file.
        :rtype: numpy.ndarray. numpy.ndarray, Path
        """
        ex = self.examples[item]
        if not self.load_into_memory:
            ex = np_load(str(ex), allow_pickle=True)

        in_e, ou_e = [ex[i].item() for i in [self.input_name, self.output_name]]
        
        path_to_audio = '/home/minkyukim/models/CLIP_AAC_implementation_project/data/clotho_audio_files/' + self.split + '/' + ex.file_name[0]
        # padding
#         path_to_audio = np.concatenate(path_to_audio, np.zeros())
        # 661500 = 15 * 44100. 15초 짜리 오디오를 44.1KHz의 SR로 샘플링 했을 때 생성되는 데이터의 개수
        audio_file, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
        audio_file = audio_file[:661500]
        
        # return in_e, ou_e, text_list
        #return in_e, ou_e, ex.file_name[0]
        return audio_file, ou_e, ex.file_name[0]

# EOF
