# 코드 소개

AudioCaps, Clotho 각 데이터셋 별로 실험에 사용했던 코드를 정리했습니다. 

<br>

## Network 세팅 

### AudioSet으로 학습된 Encoder의 Pre-trained parameter 다운받기

1. 터미널 실행 후 **AAC_Prefix/PANNs** 폴더로 이동합니다.
2. 아래 명령어를 입력합니다.
   
```
gdown https://drive.google.com/file/d/1O-rPXe_anLArvRG4Z3-nLdBYNO_JaFYL/view?usp=sharing --fuzzy
```

### Huggingface에서 제공하는 GPT2 header의 pre-trained parameter 다운받기

1. 터미널 실행 후 **ClipCap_forAAC** 폴더로 이동합니다.
2. 아래 명령어를 입력합니다.

```
gdown https://drive.google.com/file/d/15ASmIoWg0ac6qm0ixdiVwh88e8EA2MZ7/view?usp=share_link --fuzzy

```

### 논문 Table에 나오는 OURS의 parameter들 다운로드 

1. 터미널 실행 후 해당 레포지토리 경로로 이동합니다.(ex : .../AAC_Project_2022)
2. 아래 명령어를 입력합니다. 

```
gdown https://drive.google.com/file/d/1y2yeK7eO5DFY8n9l9QfiVRwv6GZLEnFA/view?usp=share_link --fuzzy
```

3. 다운로드 받은 Params_in_Table.zip을 압축해제 합니다.

<br>

## Clotho Dataset 세팅

1. 터미널 실행 후 **Clotho/clotho_audio_files** 경로로 이동합니다.
2. 아래 명령어를 입력합니다.
```
gdown https://drive.google.com/file/d/1kOuZrOs1yuOwlOky7ZohVVeiVwYQg1V0/view?usp=sharing --fuzzy
```
3. 다운로드 받은 clotho_v1.zip을 압축해제 합니다.

<br>

## AudioCaps Dataset 세팅

1. 터미널 실행 후 **AudioCaps** 경로로 이동합니다.
2. 아래 명령어를 입력합니다.

```
gdown https://drive.google.com/file/d/15ODyZmXDu_gwl-GcgQ6i_dBIeLKPG5-S/view?usp=sharing --fuzzy
```
3. 다운로드 받은 AudioCaps_Dataset.zip을 압축해제 합니다.

<br>
<br>

## Evaluation tools 다운로드

1. 터미널 실행 후 **coco_caption** 폴더로 이동합니다.
2. 아래 명령어를 입력합니다.
```
sh get_stanford_models.sh 
```

<br>

# 학습 방법 

Experiment~.py를 다음과 같이 실행하면 됩니다. 
```
# GPT2 Tokenizer를 사용하는 경우
python3 Experiment_AudioCaps.py <실험명>
python3 Experiment_Clotho.py <실험명>

# custom Tokenizer를 사용하는 경우
python3 Experiment_AudioCaps.py <실험명> <vocabulary의 크기>
python3 Experiment_Clotho.py <실험명> <vocabulary의 크기>
```

<br>

# 평가 방법

학습을 수행했다고 가정하겠습니다. 

Evaluation~.py를 다음과 같이 실행하면 됩니다. 
```
# GPT2 Tokenizer를 사용하는 경우
python3 Evaluation_AudioCaps.py <모델명> <몇 번째 epoch에서 학습시켰는지>
python3 Evaluation_Clotho.py <모델명> <몇 번째 epoch에서 학습시켰는지>

# custom Tokenizer를 사용하는 경우
python3 Evaluation_AudioCaps.py <모델명> <몇 번째 epoch에서 학습시켰는지> <vocabulary의 크기>
python3 Evaluation_Clotho.py <모델명> <몇 번째 epoch에서 학습시켰는지> <vocabulary의 크기>
```

<br>

# 추론 방법

Inference.py를 다음과 같이 실행하면 됩니다. 
```
python3 Inference.py <학습시킨 Dataset> <vocabulary 종류> <audio file의 경로>
# 예
python3 Inference.py AudioSet GPT2 ./test.wav

```
