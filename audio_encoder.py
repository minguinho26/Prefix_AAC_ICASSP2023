#AudioCLIP
from AudioCLIP.model import *

def get_audio_encoder(AudioCLIP_pretrained_path = "./AudioCLIP/model/AudioCLIP-Full-Training.pt") :
    aclip = AudioCLIP(pretrained=AudioCLIP_pretrained_path) # 학습된 AudioCLIP
    audio_encoder_in_AudioCLIP = aclip.get_audioclip()  

    return audio_encoder_in_AudioCLIP