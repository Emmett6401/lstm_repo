import pandas as pd
import pymysql
from pymysql.err import OperationalError

def connect_to_db(host_name, user_name, user_password, db_name=None, port=3306):
    """MariaDB 연결. DB가 없으면 생성하고 collation을 utf8mb4_general_ci로 설정."""
    connection = None
    try:
        connection = pymysql.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            port=port
        )
        if connection:
            cursor = connection.cursor()

            # 데이터베이스가 없으면 생성하고 collation 설정
            if db_name:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;")
                connection.select_db(db_name)  # 생성된 DB로 연결
                print(f"데이터베이스 '{db_name}'가 생성되었거나 이미 존재합니다.")
            print("MariaDB 서버에 성공적으로 연결되었습니다.")
        else:
            print("MariaDB 연결에 실패했습니다.")
    except OperationalError as e:
        print(f"Error: '{e}'")  # 구체적인 에러 메시지 출력
    
    return connection

def create_table_from_df(connection, df, table_name):
    """DataFrame으로부터 MariaDB 테이블 생성"""
    if connection is None:  # connection이 None인 경우 처리
        print(f"Error: MariaDB 연결이 설정되지 않았습니다. '{table_name}' 시트를 처리하지 못했습니다.")
        return

    cursor = connection.cursor()

    # 테이블 생성 SQL 구문
    columns = ', '.join([f'`{col}` VARCHAR(255)' for col in df.columns])  # 간단하게 모든 열을 VARCHAR로 가정
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
    {columns}
    ) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """
    cursor.execute(create_table_query)

    # 데이터 삽입 SQL 구문
    for _, row in df.iterrows():
        values = ', '.join([f"'{str(val)}'" for val in row])
        insert_query = f"INSERT INTO {table_name} VALUES ({values});"
        cursor.execute(insert_query)

    connection.commit()

def excel_to_mysql(file_path, host_name, user_name, user_password, db_name, port=3306):
    """Excel 파일의 시트를 MariaDB 테이블로 변환하여 저장"""
    connection = connect_to_db(host_name, user_name, user_password, db_name, port)

    if connection is None:  # MariaDB 연결 실패 시 처리
        print("MariaDB 연결에 실패했습니다. 프로세스를 종료합니다.")
        return

    # Excel 파일 읽기
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        # 각 시트를 DataFrame으로 변환
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        # 테이블 이름은 시트 이름을 기반으로 설정
        table_name = sheet_name

        print(f"{sheet_name} 시트를 MariaDB로 저장 중...")

        # DataFrame을 MariaDB 테이블로 변환하여 저장
        create_table_from_df(connection, df, table_name)

    if connection:
        connection.close()

# MariaDB 접속 정보
host_name = "localhost"
user_name = "display"
user_password = "0000"
db_name = "your_database"  # DB 이름
port = 3306

# Excel 파일 경로
file_path = 'predicted.xlsx'

# Excel 파일을 MariaDB로 변환하여 저장
excel_to_mysql(file_path, host_name, user_name, user_password, db_name, port)
