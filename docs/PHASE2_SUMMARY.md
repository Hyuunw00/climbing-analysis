# Phase 2 구현 요약

## 추가된 기능

### 1. 퍼포먼스 스코어 (performance_scorer.py)
- **Smoothness**: 연속 프레임 간 관절 움직임의 jerk(가속도 변화율) 기반. 낮을수록 부드러움
- **Technique**: 무게중심(COM) 이동 경로 vs 직선 거리 비율. 효율적일수록 높음
- **Stable**: 연속 프레임에서 관절 변위의 표준편차. 흔들림 적을수록 높음
- **Adhesion**: 손/발 키포인트가 홀드 바운딩 박스 내 접촉 비율
- **Live Score**: 4개 평균 (0~100)
- **Vertical Velocity**: 프레임 간 COM y축 변화 / 시간

### 2. 힘 분포 추정 (force_estimator.py)
- 무게중심(COM): 어깨/골반/무릎 가중 평균
- 4개 접촉점(왼손/오른손/왼발/오른발)에서 COM까지 거리 기반 힘 비율 계산
- 합계 항상 1.0 (100%)
- 오버레이에 원형 게이지 표시 (초록→빨강 그라데이션)
- 무게중심 위치에 흰색 십자 표시

### 3. URL 입력 (downloader.py)
- POST /api/analyze/url 엔드포인트
- yt-dlp로 유튜브/인스타 등에서 영상 다운로드
- 2분/100MB 제한 동일 적용
- 프론트에 URL 입력 필드 추가

### 4. 실시간 카메라 (realtime.py)
- WS /api/realtime WebSocket 엔드포인트
- 프론트에서 후면 카메라 스트림 → 3fps 캡처 → base64 JPEG 전송
- 서버에서 포즈/홀드/예측/힘분포 분석 → JSON 응답
- 자동 재연결 (3회), LIVE 뱃지 표시
- Upload/Camera 모드 전환 UI

### 5. 모바일 최적화
- 터치 타겟 44px 확보 (모든 버튼)
- 영상 영역: 모바일 aspect-[3/4], 태블릿 aspect-video
- URL 입력 폼 세로 스택
- 대시보드 모바일 전폭
- 음소거 토글 추가

## 데이터 흐름 (Phase 2 추가분)

```
프레임 분석 파이프라인:
  포즈 추정 → 홀드 감지 → 색상 분류 → 루트 필터링
      ↓
  Pass 2:
  ├── 퍼포먼스 스코어 (프레임 시퀀스 기반)
  ├── 힘 분포 추정 (프레임별)
  └── 경로 예측 (필터링된 홀드 기반)
      ↓
  FrameData {holds, pose, predictions, metrics, force_distribution, route_color}
```

## 파일 구조 (Phase 2 추가)

```
backend/app/services/
  ├── performance_scorer.py  # 퍼포먼스 메트릭 계산
  ├── force_estimator.py     # 힘 분포 추정
  └── ...

backend/app/routers/
  ├── realtime.py            # WebSocket 실시간 분석
  └── ...

backend/app/utils/
  ├── downloader.py          # yt-dlp 영상 다운로드
  └── ...

frontend/src/components/video/
  ├── CameraCapture.tsx      # 카메라 스트림 + WebSocket
  └── ...
```
