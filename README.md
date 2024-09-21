# Sketch 이미지 데이터 분류

## 🥇 팀 구성원
<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kimsuckhyun">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004010%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김석현</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/kupulau">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003808%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>황지은</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/lexxsh">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003955%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이상혁</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/june21a">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003793%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>박준일</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/glasshong">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004034%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>홍유리</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>

<br />

## 🗒️ 프로젝트 개요
Sketch이미지 분류 경진대회는 주어진 데이터를 활용하여 모델을 제작하고 어떤 객체를 나타내는지 분류하는 대회입니다.

Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.
<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.09.10 (화) 10:00 ~ 2024.09.26 (목) 17:00

프로젝트 세부 일정 
(수정)

## 🏆 프로젝트 결과 (수정)
- Private 리더보드에서 최종적으로 아래와 같은 결과를 얻었습니다.  


## 📁 데이터셋 구조 

```
📦data
 ┣ 📜sample_submission.csv
 ┣ 📜test.csv
 ┣ 📜train.csv
 ┣ 📂test
 ┃ ┣ 📜0.JPEG
 ┃ ┣ 📜1.JPEG
 ┃ ┣ 📜2.JPEG
 ┃ ┗ ...
 ┣ 📂train
 ┃ ┣ 📂n01443537
 ┃ ┣ 📂n01484850
 ┃ ┗ ...
```
- 학습에 사용할 이미지 데이터는 15,021개로 data/train/ 아래에 각 객체별 폴더로 구분되어 있습니다. 
- 제공되는 이미지는 주로 사람의 손으로 그려진 드로잉이나 스케치로 구성되어 있습니다. 
- train.csv와 test.csv에는 각 이미지별 폴더명(class_name), 이미지 경로(image_path), 예측해야할 class(target)에 대한 정보가 포함되어 있습니다.

<br />

## 📁 프로젝트 구조 
```
📦level1-imageclassification-cv-05-main
 ┣ 📂.github
 ┃ ┗ 📜.keep
 ┣ 📂baseline_code
 ┃ ┣ 📜baseline_code.ipynb
 ┃ ┗ 📜eda.ipynb
 ┣ 📂config
 ┃ ┣ 📜test_setting.yml
 ┃ ┣ 📜test_transform
 ┃ ┣ 📜test_transform.yml
 ┃ ┣ 📜training_setting.yml
 ┃ ┣ 📜train_transform.yml
 ┃ ┗ 📜transform.json
 ┣ 📂dataloader
 ┃ ┣ 📜dataloader.py
 ┃ ┗ 📜preprocess.py
 ┣ 📂model
 ┃ ┣ 📜_loss.py
 ┃ ┣ 📜_model.py
 ┃ ┣ 📜_optimizer.py
 ┃ ┗ 📜_schedular.py
 ┣ 📂util
 ┃ ┣ 📜seed.py
 ┃ ┣ 📜utility.py
 ┃ ┗ 📜visualize.py
 ┣ 📜.gitignore
 ┣ 📜augmentation_list.txt
 ┣ 📜infer.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┣ 📜timm_list.txt
 ┗ 📜train.py
```
### (수정)
#### 1) `dataset.py` 
- Dataset class와 Augmentation class를 구현한 파일
- CustomAugmentation, ImgaugAugmentation 구현
#### 2) `loss.py` 
- 이미지 분류에 사용될 수 있는 다양한 Loss 들을 정의한 파일
- Cross Entropy, Focal Loss, Label Smoothing Loss, F1 Loss 구현
#### 3) `model.py`
- 학습에 사용한 Model 클래스를 구현한 파일 
- resnet34, resnet50, Resnet34CategoryModel, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b5, vit_base_patch16_224, vit_B_16_imagenet1k, vit-age-classifier
#### 4) `train.py`
- config 값을 통해 학습 파라미터들을 불러오고 training 함수에 전달하는 파일
- wandb를 통한 학습 진행상황 로깅 구현
#### 5) `function.py`
- 입력으로 들어온 파라미터를 통해 train 데이터셋으로 모델 학습을 진행하는 파일
- 조건에 맞는 학습(k-fold, class별 학습)을 진행 후 best 모델을 저장
#### 6) `inference`
- `inference.py` : 학습 완료된 모델을 통해 eval set에 대한 예측 값을 구하고 이를 .csv 형식으로 저장하는 파일 
- `k-fold-inference.py` : hard voting 방식으로 k-fold를 적용하여 예측값을 구하고, 이를 .csv 형식으로 저장하는 파일
#### 7) `ensemble_weights.py`
- 여러 학습 모델들로 생성한 결과들을 hard voting하여 앙상블된 결과를 추론하는 파일
- inference 파일을 통해 생성된 .csv 파일을 입력으로 받아 새로운 .csv 파일을 생성 및 저장
#### 8) `redis_model_scheduler`
- `redis_publisher.py` : redis queue에 원하는 학습 config 값을 전달하는 파일
- `schedlue_search.py` : redis queue에서 학습을 기다리고 있는 config 값을 조회 및 삭제하는 기능 구현
- `task_consumer.py` : redis queue에 값이 들어오면 순차적으로 config를 입력받아 train에 전달

<br />

## ⚙️ requirements

- black==23.9.1
- matplotlib==3.8.2
- numpy==1.26.0
- pandas==2.1.4
- Pillow==9.4.0
- python-dotenv==1.0.0
- scikit-learn==1.2.2
- tensorboard==2.15.1
- torch==2.1.0
- torchvision==0.16.0
- timm==0.9.12
- transformers==4.36.2
- pytorch-pretrained-vit==0.0.7
- wandb==0.16.1
- redis==5.0.1
- GitPython==3.1.40
- imgaug==0.4.0
- efficientnet-pytorch ==0.7.1

`pip3 install -r requirements.txt`

<br />

## ▶️ 실행 방법

#### Train
`python train.py`

#### Inferecne
`python inference.py --model_dir [모델저장경로]`

#### scheduler를 통한 실행
`python ./redis_model_scheduler/redis_publisher.py --ip [ip주소] --port [port번호] --user [이름]`  

`python ./redis_model_scheduler/task_consumer.py --ip [ip주소] --port [port번호] --mode [mode이름]`


<br />

## ✏️ Wrap-Up Report   
- [Wrap-Up Report](https://github.com/boostcampaitech6/level1-imageclassification-cv-02/blob/main/docs/CV%EA%B8%B0%EC%B4%88%EB%8C%80%ED%9A%8C_CV_2%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(02%EC%A1%B0)_%EC%97%85%EB%A1%9C%EB%93%9C%EC%9A%A9.pdf)
