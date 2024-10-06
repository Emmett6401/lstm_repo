# 1. 베이스 이미지 설정 (파이썬 환경 설정)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정 (컨테이너 안의 디렉토리)
WORKDIR /app
# 3. 호스트 시스템(윈도우)에서 컨테이너로 파일 복사
COPY . .

# pip 업그레이드
RUN pip install --upgrade pip

# 4. 패키지 설치 (requirements.txt에서 필요한 라이브러리 설치)
RUN pip install --no-cache-dir -r requirements.txt

# 5. 컨테이너 실행 시 기본으로 실행할 명령 설정 (예: LSTM 모델 스크립트 실행)
CMD ["python", "app4_flask.py"]
