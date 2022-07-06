# AAC_Project_2022

### 데이터셋 생성
---

<br>

1. dataset을 생성합니다.
[해당 링크](https://zenodo.org/record/3490684)에 들어가서 clotho_v1 dataset을 다운받아줍니다.

![스크린샷 2022-07-06 오후 9 52 36](https://user-images.githubusercontent.com/50979281/177554471-5ddbc408-2973-4068-b2b3-5a5249402fa0.png)

2. 다운받은 파일들 중 .7z 파일들을 압축풀면 development, evaluation폴더가 나옵니다. 이 폴더들과 다운받은 .csv파일을 다음과 같이 data 폴더에 넣어줍니다.

data/ <br>
 | - clotho_audio_files/ <br>
 |   | - development/ <br>
 |   | - evaluation/ <br>
 | - clotho_csv_files/ <br>
 |   |- clotho_captions_development.csv <br>
 |   |- clotho_captions_evaluation.csv  <br>
 
 3. cmd 혹은 터미널을 실행 후 해당 프로젝트의 경로로 이동합니다.
 4. 터미널에 "python processes/dataset.py"를 입력합니다. 
 
 <img width="649" alt="스크린샷 2022-07-06 오후 10 09 52" src="https://user-images.githubusercontent.com/50979281/177557856-714d57cc-7861-434e-af41-c905794a2de0.png">

그러면 clotho dataset 'class' 가 사용하는 데이터가 생성됩니다.

### Pre_trained network 사용
사전학습된 AudioCLIP의 audio encoder를 사용하려면 .pt파일이 필요합니다. [해당 경로](https://github.com/AndreyGuzhov/AudioCLIP/releases)에서 AudioCLIP-Partial-Training.pt를 다운받으신 뒤 ACLIP/model 경로에 넣어줍니다.

![스크린샷 2022-07-06 오후 10 30 03(2)](https://user-images.githubusercontent.com/50979281/177561863-b856088b-5da2-4126-a889-e437eb514f6e.png)


### 사용 예

<img width="491" alt="스크린샷 2022-07-06 오후 10 25 31" src="https://user-images.githubusercontent.com/50979281/177560824-392ed8e0-e65a-4c71-8efe-7a1d22cfc964.png">
