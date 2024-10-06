from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')  # GUI를 사용하지 않는 백엔드로 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 설정 추가
app.secret_key = 'your_secret_key'  # 세션 관리용 비밀 키

# 로그 설정
logging.basicConfig(level=logging.DEBUG)  # DEBUG 레벨로 설정

def process_data(file_path, output_file_path):
    logging.info("데이터 처리 시작: %s", file_path)
    try:
        excel_data = pd.ExcelFile(file_path)
        logging.info("Excel 파일 읽기 완료.")
    except Exception as e:
        logging.error(f"Error reading the Excel file: {e}")
        raise ValueError("Excel 파일을 읽는 중 오류가 발생했습니다. 파일 형식을 확인하세요.")
    
    processed_data = {}

    for sheet_name in excel_data.sheet_names:
        logging.info("처리 중인 시트: %s", sheet_name)
        df = excel_data.parse(sheet_name)
        df.columns = df.columns.str.strip()

        # '년월일' 열을 문자열로 변환 후 날짜 처리
        if '년월일' in df.columns:
            df['년월일'] = df['년월일'].astype(str).str.replace('년', '-').str.replace('월', '-').str.replace('일', '')
            df['년월일'] = pd.to_datetime(df['년월일'], errors='coerce')
        if '예측일자' in df.columns:
            df['예측일자'] = df['예측일자'].astype(str).str.replace('년', '-').str.replace('월', '-').str.replace('일', '')
            df['예측일자'] = pd.to_datetime(df['예측일자'], errors='coerce')

        numeric_cols = ['유입량(㎥/일)', '인구수(명)', '강수량(mm)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '년월' not in df.columns and '년월일' in df.columns:
            df['년월'] = df['년월일'].dt.to_period('M')

        if '인구수(명)' in df.columns:
            growth_rate = 0.02
            df['인구수(명)'] *= (1 + growth_rate)

        if '강수량(mm)' in df.columns and '년월일' in df.columns:
            df['월'] = df['년월일'].dt.month
            df['일'] = df['년월일'].dt.day
            df['강수량(mm)'] = df.groupby(['월', '일'])['강수량(mm)'].transform(lambda x: x.fillna(x.mean()))

        df.drop(columns=['월', '일'], inplace=True)
        processed_data[sheet_name] = df

    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in processed_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info("저장된 시트: %s", sheet_name)

def create_and_train_lstm_model(X_train, y_train, time_steps, epoch, batch_size, lstm_units, dropout_rate):
    logging.info("LSTM 모델을 생성하고 훈련합니다.")
    
    model = Sequential()
    model.add(Input(shape=(time_steps, X_train.shape[2])))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
    
    logging.info("LSTM 모델 훈련 완료.")
    return model

def prepare_data_for_lstm(df, time_steps, future_days):
    logging.info("LSTM 데이터 준비 중...")
    features = ['강수량(mm)', '인구수(명)']
    target = '유입량(㎥/일)'
    
    feature_data = df[features].values
    target_data = df[target].values.reshape(-1, 1)
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(feature_data)
    scaled_target = target_scaler.fit_transform(target_data)
    
    X = []
    y = []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i-time_steps:i, :])
        y.append(scaled_target[i])
    X, y = np.array(X), np.array(y)
    
    logging.info("LSTM 데이터 준비 완료. 데이터 포맷: X=%s, y=%s", X.shape, y.shape)
    return X, y, feature_scaler, target_scaler

def process_sheets_for_lstm(file_path, output_path, model_dir, output_graph_dir, epoch, batch_size, lstm_units, dropout_rate):
    all_sheets = pd.ExcelFile(file_path).sheet_names
    logging.info("시트 수: %d", len(all_sheets))
    progress = []

    for sheet_name in all_sheets:
        logging.info("Processing sheet: %s", sheet_name)  # 어떤 시트를 처리하는지 로그 추가
        progress.append(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        if '년월일' in df.columns:
            df['년월일'] = pd.to_datetime(df['년월일'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')
            if df['년월일'].isnull().all():
                logging.warning(f"'{sheet_name}' 시트에 유효한 '년월일' 값이 없습니다. 스킵합니다.")
                continue
        else:
            logging.warning(f"'{sheet_name}' 시트에 '년월일' 열이 없습니다. 스킵합니다.")
            continue

        df.sort_values('년월일', inplace=True)

        if '유입량(㎥/일)' in df.columns:
            df['유입량(㎥/일)'] = df['유입량(㎥/일)'].fillna(0)

        time_steps = 30  
        future_days = 30  
        X, y, feature_scaler, target_scaler = prepare_data_for_lstm(df, time_steps, future_days)

        logging.info("Training LSTM model for sheet: %s", sheet_name)  # 훈련 시작 로그 추가
        model = create_and_train_lstm_model(X, y, time_steps, epoch, batch_size, lstm_units, dropout_rate)

        model_save_path = os.path.join(model_dir, f'{sheet_name}_lstm_model.keras')
        model.save(model_save_path)
        logging.info(f"'{sheet_name}' 시트의 모델을 저장했습니다: {model_save_path}")

        # 예측 결과 저장 및 그래프 저장
        df_predictions = predict_existing_data(df.copy(), model, feature_scaler, target_scaler, time_steps)
        plot_and_save_graphs(df_predictions, output_graph_dir, sheet_name)

        logging.info("Completed processing for sheet: %s", sheet_name)  # 시트 처리 완료 로그 추가
        progress.append(f"Completed processing for sheet: {sheet_name}")

    logging.info("모든 시트 처리 완료. 끝났습니다.")
    progress.append("모든 시트 처리 완료. 끝났습니다.")
    return progress

def predict_existing_data(df, model, feature_scaler, target_scaler, time_steps):
    logging.info("기존 데이터를 기반으로 예측합니다.")
    features = ['강수량(mm)', '인구수(명)']
    target = '유입량(㎥/일)'

    feature_data = df[features].values
    target_data = df[target].values.reshape(-1, 1)

    scaled_features = feature_scaler.transform(feature_data)
    scaled_target = target_scaler.transform(target_data)

    X = []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i-time_steps:i, :])
    X = np.array(X)

    predictions_scaled = model.predict(X)
    predictions = target_scaler.inverse_transform(predictions_scaled)

    df['예측 유입량(㎥/일)'] = np.nan
    df.loc[time_steps:, '예측 유입량(㎥/일)'] = predictions.flatten()
    df['예측일자'] = df['년월일']

    # 예측 유입량 NaN 값 처리: 이전 값으로 채움
    df['예측 유입량(㎥/일)'] = df['예측 유입량(㎥/일)'].ffill()

    return df

def plot_and_save_graphs(df, output_graph_dir, sheet_name):
    # 그래프 저장 디렉터리 확인 및 생성
    if not os.path.exists(output_graph_dir):
        os.makedirs(output_graph_dir)

    # 한글 글꼴 설정
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 운영 체제에 따라 경로를 조정
    font_prop = fm.FontProperties(fname=font_path, size=14)
    plt.rc('font', family=font_prop.get_name())

    plt.figure(figsize=(14, 7))
    plt.plot(df['년월일'], df['유입량(㎥/일)'], label='Actual Inflow', color='blue')
    plt.plot(df['년월일'], df['예측 유입량(㎥/일)'], label='Predicted Inflow', color='orange')
    
    plt.xlabel('Date')
    plt.ylabel('Inflow (㎥/일)')
    plt.title(f'{sheet_name} - Inflow Prediction vs Actual')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    graph_path = os.path.join(output_graph_dir, f'{sheet_name}_예측_그래프.png')
    plt.savefig(graph_path, format='png')  # 그래프 저장
    plt.close()  # 그래프 창 닫기

@app.route('/')
def home():
    return render_template('train_predict.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        epoch = int(request.form['epoch'])
        batch_size = int(request.form['batch'])
        lstm_units = int(request.form['lstm_units'])
        dropout_rate = float(request.form['dropout'])

        file_path = '예측된_수정된_취합_일자_강수량_인구_유입량_20240829_제공.xlsx'  # 예시로 하드코딩된 파일 경로
        output_path = '예측결과.xlsx'
        model_dir = 'models'
        output_graph_dir = 'graphs'

        # 데이터 처리
        process_data(file_path, output_path)
        
        # 시트별로 LSTM 모델 학습 및 예측 수행
        progress = process_sheets_for_lstm(file_path, output_path, model_dir, output_graph_dir, epoch, batch_size, lstm_units, dropout_rate)

        logging.info("모델 학습 및 예측 완료.")
        return jsonify({'status': 'success', 'progress': progress})
    except Exception as e:
        logging.error(f"오류 발생: {e}")  # 오류 메시지 출력
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
