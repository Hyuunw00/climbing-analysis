# Phase 3 구현 요약

## 목표
ML 기반 경로 예측 + 홀드 감지 정확도 개선

## 구현 완료

### 1. ML 경로 예측 파이프라인 (코드 완료, 학습 대기)
- **features.py**: 포즈+홀드 → 33개 feature 추출 (거리, 각도, COM, 리치, 체형 등)
- **labeler.py**: 영상에서 move event 자동 감지 → 학습 샘플 생성
- **predictor.py**: XGBoost 모델 로딩 + 추론 (MLPathPredictor)
- **path_predictor.py**: ML 우선 + 규칙 기반 fallback
- **train_model.py**: XGBoost 학습 스크립트 (Colab 호환)
- **generate_training_data.py**: 영상 → 라벨링 → Parquet
- **유저 영상 자동 수집**: 분석 시 move event → user_training_data.jsonl 자동 저장

### 2. 접촉 기반 루트 감지 (CVPR 2025 "The Way Up" 방식)
- 이전: HSV 색상 투표 (fps 의존적, 불안정)
- 변경: 포즈 키포인트 × 홀드 bbox 겹침으로 접촉 감지
- 손가락(index) / 발끝(foot_index) 1초 이상 연속 접촉 → "사용 홀드"
- 접촉 홀드 색상 투표 → 루트 색상 결정 → 같은 색 홀드 필터링

### 3. 홀드 감지 모델 파인튜닝
- **colab_train.py**: 원클릭 학습 스크립트 (Drive 마운트 → 데이터 병합 → YOLOv8s 학습)
- 데이터셋: Neural Climb V2 (5,854장) + Climbing Holds and Volumes (1,717장)
- 모델: YOLOv8n (nano) → YOLOv8s (small) 업그레이드
- 상태: Colab T4 GPU에서 학습 진행 중

### 4. 들찍 영상 대응
- 5프레임마다 YOLO 홀드 재감지 (카메라 이동 추적)
- 이전: 초반 홀드맵 고정 → 들찍 시 홀드 위치 불일치
- 변경: 주기적 재감지로 홀드 위치 실시간 업데이트

### 5. 포즈 안정성 개선
- MediaPipe lite → **heavy** 모델 교체 (정확도 향상)
- 포즈 실패 시 마지막 성공 포즈 유지 (최대 3프레임)
- 평균 visibility < 0.4 시 유령 포즈 필터링

### 6. Dab 감지
- 접촉 홀드 중 루트 색상이 아닌 것 = dab
- 프론트엔드에서 빨간 박스 + "DAB" 라벨로 표시

### 7. 분석 속도 개선
- Phase 2+3 포즈 캐싱 (포즈 추정 1회만 실행)
- frame_rate: 2fps → 5fps (다이노/코디 대응)

### 8. UI 개선
- 분석 진행 UI: 0%부터 퍼센트 표시 + 단계별 라벨
- 볼륨 슬라이더 추가
- 영상 full 채우기 (object-cover)

## 미완료 (학습 필요)

### ML 경로 예측 학습
- 코드 준비 완료, 홀드 감지 모델 학습 후 진행
- 완등 영상 URL 수집 → generate_training_data.py → train_model.py
- XGBoost: Colab 무료 티어에서 학습 가능

## 현재 한계점
- 홀드 감지 정확도: nano 모델 한계 (학습 완료 시 개선 예상)
- 포즈 추정: 클라이머 멀리/위에서 촬영 시 부정확 (MediaPipe 범용 모델 한계)
- 분석 속도: heavy 포즈 + 홀드 재감지로 느려짐 (최적화 필요)
- 볼륨 인식: 현재 모델로는 볼륨 감지 약함 (학습 모델로 개선 예상)
