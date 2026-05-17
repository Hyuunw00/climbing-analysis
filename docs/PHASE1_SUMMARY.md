# Phase 1 구현 요약

## 아키텍처

```
[프론트엔드 Next.js]  ←→  [백엔드 FastAPI]
   영상 업로드              영상 분석
   오버레이 렌더링          포즈 추정
   대시보드 UI             홀드 감지
   상태 관리               경로 예측
```

## 전체 데이터 흐름

```
1. 영상 업로드
   사용자 → VideoUploader → POST /api/analyze → 파일 저장 + job_id 반환

2. 분석 (백그라운드)
   프레임 추출 (5fps) → 각 프레임마다:
     ├── MediaPipe 포즈 추정 (33개 관절)
     ├── YOLOv8 홀드 감지 (⚠ 모델 미학습)
     └── 규칙 기반 경로 예측 (홀드 데이터 필요)

3. 결과 렌더링
   프론트 1초 간격 폴링 → completed 시 결과 로드 →
   영상 재생 시 현재 시간에 맞는 프레임 데이터 →
     ├── Canvas 오버레이 (스켈레톤, 홀드 박스, 예측 선)
     └── 대시보드 (메트릭 바, 스코어)
```

## 프론트엔드 구현

### 상태 머신 (page.tsx)
```
idle → uploading → analyzing → ready → (New Video) → idle
                                ↓
                              error → (Try again) → idle
```
- localStorage로 job_id 저장 → 새로고침 시 세션 복원
- 1초 폴링으로 분석 진행률 추적

### 오버레이 렌더링 (OverlayRenderer.tsx)
- Canvas를 video 위에 absolute로 겹침
- 레터박스/필러박스 오프셋 자동 계산 (세로 영상 지원)
- 그리는 것:
  - 🟢 클라이머 바운딩 박스 + "CLIMBER" 라벨
  - 🟢 포즈 스켈레톤 (12개 뼈대 연결)
  - 🔵 홀드 바운딩 박스 (현재 모델 없어서 비활성)
  - 🟡 경로 예측 점선 (홀드 없어서 비활성)

### 대시보드 (DashboardPanel.tsx)
- Route Color, Vertical Velocity
- Live Performance 4개 게이지 바 (Smoothness, Technique, Stable, Adhesion)
- Live Score / Avg Score + 등급 (A~D)
- frameData prop으로 실시간 업데이트

### API 클라이언트 (api.ts)
- `uploadVideo(file)` → POST /api/analyze
- `getAnalysisStatus(jobId)` → GET /api/analyze/{job_id}
- `getAnalysisResult(jobId)` → GET /api/analyze/{job_id}/result
- `getVideoUrl(jobId)` → 영상 스트리밍 URL
- snake_case ↔ camelCase 자동 매핑

## 백엔드 구현

### API 엔드포인트 (analysis.py)
| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | /api/analyze | 영상 업로드 + 분석 시작 |
| GET | /api/analyze/{job_id} | 분석 상태 조회 |
| GET | /api/analyze/{job_id}/result | 분석 결과 조회 |
| GET | /api/analyze/{job_id}/video | 업로드 영상 서빙 |
| GET | /api/health | 헬스체크 |

- BackgroundTasks로 비동기 분석
- in-memory dict로 job 상태 관리 (서버 재시작 시 소실)
- 파일 검증: mp4/mov/webm, 100MB, 2분 제한

### 포즈 추정 (pose_estimator.py)
- MediaPipe PoseLandmarker Tasks API
- 33개 관절 좌표 (x, y, z, visibility)
- 모델 자동 다운로드 (pose_landmarker_lite.task)
- 실패 시 빈 리스트 반환

### 홀드 감지 (hold_detector.py)
- YOLOv8 기반 구조
- ⚠ **학습된 모델 없음** → 항상 빈 결과 반환
- 모델 파일 있으면 자동 로딩 (graceful degradation)

### 경로 예측 (path_predictor.py)
- 규칙 기반: 거리 × 방향가중치 (위쪽 1.5배)
- 리치 범위(0.35) 밖 제외
- 손당 상위 3개 후보 반환
- ⚠ **홀드 감지 없으면 예측 불가**

### 영상 처리 (video_processor.py)
- OpenCV로 프레임 추출 (설정 fps로 다운샘플링)
- 각 프레임마다 포즈 → 홀드 → 예측 순서로 파이프라인 실행

## 현재 동작하는 것 vs 안 하는 것

### ✅ 동작
- 영상 업로드 + 재생
- 클라이머 포즈 트래킹 (초록 박스 + 스켈레톤)
- 새로고침 시 세션 유지
- New Video 버튼으로 초기화
- 대시보드 UI (포즈 데이터 기반 메트릭)

### ⚠ 부분 동작
- 대시보드 메트릭이 포즈 신뢰도 기반 근사값 (실제 분석 로직 아님)
- Vertical Velocity 항상 0 (멀티프레임 계산 미구현)

### ❌ 미동작 (모델 필요)
- 홀드 감지 → 학습된 모델 파일 필요
- 경로 예측 → 홀드 데이터에 의존
- 홀드 바운딩 박스 오버레이
- 예측 경로 점선 오버레이

## 기술 스택

| 영역 | 기술 |
|------|------|
| 프론트 | Next.js 19, TypeScript, Tailwind CSS, Canvas API |
| 백엔드 | FastAPI, Python 3.13, Pydantic v2 |
| AI | MediaPipe PoseLandmarker, YOLOv8 (ultralytics) |
| 영상 | OpenCV, FFmpeg |
| 배포 | 로컬 (localhost:3000 + localhost:8000) |
