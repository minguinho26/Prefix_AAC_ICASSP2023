# 코드 소개

AudioCaps, Clotho 각 데이터셋 별로 실험에 사용했던 코드를 정리했습니다. 

<br>

## Network 세팅 

1. 터미널 실행 후 **ClipCap_forAAC/PANNs** 폴더로 이동합니다.
2. 아래 명령어를 입력합니다.
   
```
gdown https://drive.google.com/file/d/1O-rPXe_anLArvRG4Z3-nLdBYNO_JaFYL/view?usp=sharing --fuzzy

```

3. **ClipCap_forAAC** 폴더로 이동합니다.
4. 아래 명령어를 입력합니다.
```
gdown https://drive.google.com/file/d/15ASmIoWg0ac6qm0ixdiVwh88e8EA2MZ7/view?usp=sharing --fuzzy

```

5. **ClipCap_forAAC/pre_trained_params_from_audiocaps** 폴더로 이동합니다.
6. 아래 명령어를 입력합니다.
```
gdown https://drive.google.com/file/d/1VK2mCuBgICG2Ckt9PFNS_r0-QQYZUlJk/view?usp=sharing --fuzzy

```
7. 다운로드 받은 Pre_trained_params.zip을 압축해제 합니다.

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

## 기타 세팅

1. AudioCaps, ClipCap_forAAC, Clotho 폴더가 있는 루트 경로로 이동합니다
2. 아래 명령어를 입력합니다.
```
gdown https://drive.google.com/file/d/1RK-qCJ5UM9sPl5Nh4PCrq3n8GLl1qXmW/view?usp=sharing --fuzzy
```
3. 다운로드 받은 coco_caption.zip을 압축해제 합니다.

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
