#AudioCLIP
from esresnet.fbsp import *
import torch

class Audio_Encoder(ESResNeXtFBSP) :
    def __init__(self):

        super(Audio_Encoder, self).__init__(
            n_fft=2048,
            hop_length=561,
            win_length=1654,
            window='blackmanharris',
            normalized=True,
            onesided=True,
            spec_height=-1,
            spec_width=-1,
            num_classes=1024,
            apply_attention=True,
            pretrained=False
        )

def get_audio_encoder(pretrained_path = "./esresnet/pre_trained_params.pt") :
    
    audio_encoder = Audio_Encoder()
    
    audio_encoder.load_state_dict(torch.load(pretrained_path))
    
    return audio_encoder