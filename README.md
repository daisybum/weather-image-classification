# 날씨 주행 데이터셋 분류 프로젝트

날씨 주행 데이터셋을 활용한 멀티 GPU 학습 구현체입니다. 본 프로젝트는 MobileNetV3를 사용하여 주행 상황에서의 다양한 날씨 조건을 효율적으로 분류합니다.

## 🌟 주요 기능

- DistributedDataParallel (DDP)을 활용한 멀티 GPU 학습 지원
- MobileNetV3 백본과 커스텀 분류기 구현
- 효율적인 데이터 로딩 및 처리
- 포괄적인 로깅 시스템
- 자동 체크포인트 저장
- 다중 GPU 분산 학습 지원

## 📋 필요 라이브러리

```bash
torch>=2.0.0
torchvision>=0.15.0
lion-pytorch
pillow
tqdm
```

## 🗂 프로젝트 구조

```
.
├── checkpoints/             # 모델 체크포인트
├── utils/
│   └── transform.py        # 데이터 변환 유틸리티
├── model_trainer.py        # 주요 학습 로직
└── main.py                # 학습 실행 스크립트
```

## 🚀 빠른 시작

1. **환경 설정**
   ```bash
   pip install -r requirements.txt
   ```

2. **데이터셋 준비**
   - 데이터셋을 다음과 같은 구조로 배치합니다:
   ```
   dataset/
   ├── Training/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   └── Validation/
       ├── class1/
       ├── class2/
       └── ...
   ```

3. **학습 시작**
   ```bash
   python main.py
   ```

## 💻 멀티 GPU 학습

본 구현체는 다중 GPU 분산 학습을 지원합니다. 코드는 자동으로:
- 가용 GPU에 데이터 분배
- 학습 중 그래디언트 동기화
- 마스터 GPU에서 체크포인트 저장을 관리

GPU 지정 방법:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'  # GPU 2,3,4,5 사용
```

## 🔧 설정

`main.py`의 주요 설정 옵션:

```python
# 데이터셋 설정
train_path = "D:/4type_weather_driving_dataset/Training"
valid_path = "D:/4type_weather_driving_dataset/Validation"
batch_size = 64
num_workers = 8

# 학습 설정
num_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-2

# 모델 설정
num_classes = 5
```

## 📝 로깅

학습 과정은 콘솔과 파일에 모두 기록됩니다:
- 학습/검증 손실
- 정확도 지표
- GPU 할당 정보
- 오류 메시지 및 디버깅 정보

로그는 `training.log`에 저장됩니다.

## 💾 체크포인트

검증 손실이 개선될 때마다 모델 체크포인트가 자동으로 저장됩니다. 각 체크포인트는 다음을 포함합니다:
- 모델 상태 사전
- 옵티마이저 상태
- 스케줄러 상태
- 학습 에포크 정보

## 🤝 기여하기

개선사항에 대한 이슈나 풀 리퀘스트는 언제나 환영합니다.

## 📜 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📚 참고 자료

- [MobileNetV3 논문](https://arxiv.org/abs/1905.02244)
- [PyTorch 분산 학습 튜토리얼](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)