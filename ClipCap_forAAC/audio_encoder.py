#AudioCLIP
from .esresnet.fbsp import *
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
        
    def forward(self,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self._forward_pre_processing(x)
        audio_feature = self._forward_features(x)
        reduced_audio_feature = self._forward_reduction(audio_feature) # 2048-dim(classification)
        semantic_feature = self._forward_classifier(reduced_audio_feature) # 1024-dim(embedding vector)
        
        # AudioCLIP 클래스에는 이게 있던데 audio encoder에는 이런 normalization이 없어서 추가해줌
        semantic_feature = semantic_feature / semantic_feature.norm(dim=-1, keepdim=True) 
        
        return audio_feature[0], semantic_feature

def get_audio_encoder(pretrained_path = "./ClipCap_forAAC/esresnet/pre_trained_params.pt") :
    
    audio_encoder = Audio_Encoder()
    
    audio_encoder.load_state_dict(torch.load(pretrained_path))
    
    return audio_encoder