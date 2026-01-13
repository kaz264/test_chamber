# 스마트팜 SAC 에이전트

식물재배기(스마트팜) 시뮬레이터와 SAC(Soft Actor-Critic) 강화학습 에이전트 구현

## 개요

이 프로젝트는 온도와 습도를 제어하는 스마트팜 환경의 Digital Twin을 구축하고, SAC 알고리즘으로 최적 제어 정책을 학습합니다.

## 주요 기능

- **환경 시뮬레이션**: Gymnasium 기반 스마트팜 환경
  - 상태: [현재온도, 현재습도, 목표온도, 목표습도]
  - 행동: [냉난방 출력(-1~1), 가습/제습 출력(-1~1)]
  - 물리 법칙: 자연 복귀력, 환경 노이즈 포함

- **SAC 에이전트**: Stable-Baselines3 구현
  - 연속 행동 공간 최적화
  - 엔트로피 최대화를 통한 탐험/이용 균형
  - 에너지 효율성을 고려한 보상 함수

## Google Colab에서 실행

### 학습 실행
```python
# 1. 저장소 클론
!git clone https://github.com/kaz264/test_chamber.git
%cd test_chamber

# 2. 필요한 패키지 설치
!pip install gymnasium stable-baselines3 numpy shimmy matplotlib

# 3. 노트북 실행
# test_chamber.ipynb 파일을 열어서 실행
```

### 학습된 모델 테스트
학습이 완료된 후 `test_model.ipynb`를 실행하여 다음을 테스트할 수 있습니다:
- 시각화된 제어 성능 그래프
- 다중 에피소드 통계 분석
- 극단적인 초기 조건 테스트
- 랜덤 정책과 성능 비교
- 목표값 변경 시 적응력 테스트

## 로컬 실행

```bash
# 패키지 설치
pip install gymnasium stable-baselines3 numpy shimmy

# Jupyter Notebook 실행
jupyter notebook test_chamber.ipynb
```

## 학습 결과

학습된 모델은 `sac_smartfarm_agent.zip` 파일로 저장되며, 다음과 같이 불러올 수 있습니다:

```python
from stable_baselines3 import SAC
model = SAC.load("sac_smartfarm_agent")
```

## 요구사항

- Python 3.8+
- gymnasium
- stable-baselines3
- numpy
- shimmy

## 라이선스

MIT License
