# lstm_repo
이 레포는 구경모박사_디비비젼_하수유입모델의 배포를 위함입니다.

## 주요 기능 및 프로그램 
1. 데이터 전처리 app3
2. 인구수와 강수량을 독립변수로 사용 하여 모델을 학습 app27
3. 모든 사업소별 개별 모델을 도출
4. 앙상블 learning으로 편향성 제거를 위해 통합 모델을 도출
5. 예측 결과를 시각화 app4

## 참고 예제
1. UI가 적용된 모델 학습 app99_uiSample_train_predict.py
2. 엑셀파일을 DB에 넣기 app99_dbSample_excel2DB.py
3. 각 사업소별 모델을 추가 데이터로 업뎃 하기 transfer learning app81_

## 참고 자료 
1. 모델Guide.docx
2. 배포Guide2024_10_04.pptx
3. Docker를 이용한 배포(1).pptx


## 데이터세트와 생성 파일 
1. 취합_일자_강수량_인구_유입량_20240829_제공.xlsx : 제공 받은 원시 데이터 파일
2. Prepared.xlsx : app3을 통해 데이터 전처리 된 파일 결측치 처리등
3. predicted.xlsx : app27을 통해 출력된 예측치
4. metrics.xlsx : 모델의 metrics 파일 - 편향성으로 loss율 이외에 의미 없음
5. Dockerfile : 도커 이미지 생성을 위한 도커파일 
6. requirements.txt : 개발환경 구축을 위한 package 파일

폴더 설명 
1. graph : 각 사업소별 예측 그래프
2. models: 각 사업소별 모델과 통합 모델
3. templates : flask에서 사용하는 html 파일 
