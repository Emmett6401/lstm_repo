import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging


# 기존 모델 불러오기
model_path = 'models/가좌_lstm_model.keras'
model = load_model(model_path)

# 추가 데이터를 준비하는 함수
def prepare_additional_data(df, feature_scaler, target_scaler, time_steps):
    features = ['강수량(mm)', '인구수(명)']
    target = '유입량(㎥/일)'
    
    feature_data = df[features].values
    target_data = df[target].values.reshape(-1, 1)
    
    scaled_features = feature_scaler.transform(feature_data)
    scaled_target = target_scaler.transform(target_data)
    
    X = []
    y = []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i-time_steps:i, :])
        y.append(scaled_target[i])
    X, y = np.array(X), np.array(y)
    
    return X, y


# 추가 학습 데이터 로드 및 준비
additional_data = pd.read_excel('추가_데이터.xlsx')  # 2024년 08월 01일 ~ 12월 31일의 데이터
X_additional, y_additional = prepare_additional_data(additional_data, feature_scaler, target_scaler, time_steps)

# 추가 학습
model.fit(X_additional, y_additional, epochs=10, batch_size=32, verbose=1)

# 모델 저장 (재학습된 모델을 덮어쓰거나 다른 이름으로 저장 가능)
model.save('models/가좌_lstm_model_retrained.keras')
