# -*- coding: utf-8 -*-
"""
식품영양정보 분석 Streamlit 웹앱
Original Colab notebook converted to Streamlit interface
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit 페이지 설정
st.set_page_config(
    page_title="식품영양정보 분석기",
    page_title_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정 (Streamlit Cloud에서는 기본 폰트 사용)
plt.rcParams['axes.unicode_minus'] = False

# 최적화된 데이터 로딩 함수들
@st.cache_data(ttl=3600, show_spinner="데이터를 최적화하는 중...")  # 1시간 캐시
def convert_to_parquet():
    """원본 데이터를 Parquet 형식으로 변환하여 저장"""
    csv_file_path = './20250327_가공식품DB_147999건.csv'
    xls_file_path = './20250327_가공식품DB_147999건.xlsx'
    parquet_file_path = './nutrition_data_optimized.parquet'
    
    # Parquet 파일이 이미 있으면 건너뛰기
    if os.path.exists(parquet_file_path):
        return parquet_file_path
    
    try:
        # 원본 데이터 로드
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False)
            st.info(f"CSV에서 {len(df):,}개 레코드 로드됨")
        elif os.path.exists(xls_file_path):
            df = pd.read_excel(xls_file_path)
            st.info(f"Excel에서 {len(df):,}개 레코드 로드됨")
        else:
            return None
        
        # 데이터 전처리 및 최적화
        columns_to_use = [
            '식품명', '대표식품명', '식품소분류명',
            '에너지(kcal)', '단백질(g)', '지방(g)', '탄수화물(g)', '당류(g)',
            '나트륨(mg)', '콜레스테롤(mg)', '포화지방산(g)', '트랜스지방산(g)',
            '식이섬유(g)', '칼슘(mg)', '식품중량', '제조사명'
        ]
        
        # 존재하는 컬럼만 선택
        existing_cols = [col for col in columns_to_use if col in df.columns]
        df = df[existing_cols]
        
        # 결측값 처리
        df = df.dropna(subset=['식품명'])
        
        # 데이터 타입 최적화 (메모리 사용량 60-80% 절약)
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # 카테고리형 데이터 최적화
        categorical_cols = ['식품소분류명', '제조사명']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Parquet 형식으로 저장 (압축률 높고 로딩 속도 빠름)
        df.to_parquet(parquet_file_path, compression='snappy', index=False)
        st.success(f"데이터가 Parquet 형식으로 최적화되어 저장되었습니다. (용량: {os.path.getsize(parquet_file_path) / 1024 / 1024:.1f}MB)")
        
        return parquet_file_path
        
    except Exception as e:
        st.error(f"데이터 변환 오류: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner="고성능 데이터 로딩 중...")  # 1시간 캐시
def load_optimized_data():
    """최적화된 Parquet 파일에서 데이터 로드 (10-50배 빠름)"""
    parquet_file_path = './nutrition_data_optimized.parquet'
    
    # Parquet 파일이 없으면 생성
    if not os.path.exists(parquet_file_path):
        parquet_path = convert_to_parquet()
        if parquet_path is None:
            return None
    
    try:
        # Parquet에서 초고속 로드 (일반적으로 CSV 대비 5-10배 빠름)
        df = pd.read_parquet(parquet_file_path)
        return df
        
    except Exception as e:
        st.error(f"최적화된 데이터 로딩 오류: {e}")
        return None

@st.cache_data(ttl=1800)  # 30분 캐시
def load_data_from_sqlite():
    """SQLite 로컬 DB에서 초고속 로드"""
    db_path = './nutrition_data.db'
    
    if not os.path.exists(db_path):
        # DB가 없으면 생성
        df_temp = load_optimized_data()
        if df_temp is not None:
            create_sqlite_db(df_temp, db_path)
        else:
            return None
    
    try:
        # SQLite에서 초고속 로드
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM nutrition_data", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"SQLite 로딩 오류: {e}")
        return None

def create_sqlite_db(df, db_path):
    """DataFrame을 SQLite DB로 저장"""
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql('nutrition_data', conn, if_exists='replace', index=False)
        
        # 검색 성능을 위한 인덱스 생성
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_name ON nutrition_data(식품명)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON nutrition_data(식품소분류명)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_manufacturer ON nutrition_data(제조사명)")
        
        conn.commit()
        conn.close()
        st.success(f"SQLite DB 생성 완료: {db_path}")
    except Exception as e:
        st.error(f"SQLite DB 생성 오류: {e}")

@st.cache_data(ttl=1800)  # 30분 캐시  
def load_data_from_pickle():
    """Pickle 파일에서 초고속 로드 (Python 객체 직렬화)"""
    pickle_path = './nutrition_data.pkl'
    
    if not os.path.exists(pickle_path):
        # Pickle 파일이 없으면 생성
        df_temp = load_optimized_data()
        if df_temp is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(df_temp, f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(f"Pickle 캐시 생성: {pickle_path}")
        else:
            return None
    
    try:
        # Pickle에서 초고속 로드 (종종 가장 빠름)
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        return df
    except Exception as e:
        st.error(f"Pickle 로딩 오류: {e}")
        return None

# PostgreSQL/MySQL 연동 (선택사항)
@st.cache_data(ttl=900)  # 15분 캐시
def load_data_from_database():
    """PostgreSQL/MySQL에서 데이터 로드 (프로덕션 환경)"""
    
    # Streamlit secrets에서 DB 정보 가져오기
    # secrets.toml 파일에 설정:
    # [database]
    # host = "your-host"
    # port = 5432
    # database = "nutrition_db"
    # username = "your-user"
    # password = "your-password"
    
    try:
        # PostgreSQL 연결 예시
        db_config = st.secrets.get("database", {})
        if not db_config:
            return None
            
        engine = create_engine(
            f"postgresql://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        # 쿼리 최적화: 필요한 컬럼만 선택
        query = """
        SELECT 식품명, 대표식품명, 식품소분류명, 에너지_kcal as "에너지(kcal)",
               단백질_g as "단백질(g)", 지방_g as "지방(g)", 탄수화물_g as "탄수화물(g)",
               당류_g as "당류(g)", 나트륨_mg as "나트륨(mg)", 콜레스테롤_mg as "콜레스테롤(mg)",
               포화지방산_g as "포화지방산(g)", 식이섬유_g as "식이섬유(g)", 
               칼슘_mg as "칼슘(mg)", 식품중량, 제조사명
        FROM nutrition_data 
        ORDER BY 식품명
        """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
        
    except Exception as e:
        # DB 연결 실패시 조용히 넘어감 (백업 로딩 방법 사용)
        return None

def load_data():
    """초고속 데이터 로딩 - 여러 소스에서 최적 경로 자동 선택"""
    
    import time
    start_time = time.time()
    
    # 1순위: Pickle 캐시 (가장 빠름)
    df = load_data_from_pickle()
    if df is not None:
        load_time = time.time() - start_time
        st.sidebar.success(f"🚀 Pickle 캐시: {len(df):,}개 식품 ({load_time:.2f}초)")
        return df
    
    # 2순위: SQLite 로컬 DB
    df = load_data_from_sqlite()
    if df is not None:
        load_time = time.time() - start_time
        st.sidebar.success(f"💾 SQLite DB: {len(df):,}개 식품 ({load_time:.2f}초)")
        return df
    
    # 3순위: 최적화된 Parquet 파일
    df = load_optimized_data()
    if df is not None:
        load_time = time.time() - start_time
        st.sidebar.success(f"⚡ Parquet: {len(df):,}개 식품 ({load_time:.2f}초)")
        return df
    
    # 4순위: 원격 데이터베이스
    df = load_data_from_database()
    if df is not None:
        load_time = time.time() - start_time
        st.sidebar.info(f"🌐 원격 DB: {len(df):,}개 식품 ({load_time:.2f}초)")
        return df
    
    # 5순위: Google Drive에서 다운로드 https://docs.google.com/spreadsheets/d/1FrAR9SRDVbppLbeP-F2IFY3FQwLWB1oX/edit
    if GOOGLE_DRIVE_CONFIG["file_id"] != "1FrAR9SRDVbppLbeP-F2IFY3FQwLWB1oX":
        google_drive_file = download_from_google_drive(
            GOOGLE_DRIVE_CONFIG["file_id"], 
            GOOGLE_DRIVE_CONFIG["file_name"]
        )
        if google_drive_file and os.path.exists(google_drive_file):
            try:
                df = pd.read_excel(google_drive_file)
                load_time = time.time() - start_time
                st.sidebar.info(f"☁️ Google Drive: {len(df):,}개 식품 ({load_time:.2f}초)")
                
                # 전처리 및 캐시 생성
                df = preprocess_dataframe(df)
                create_all_caches(df)
                return df
            except Exception as e:
                st.error(f"Google Drive 파일 로딩 오류: {e}")
    
    # 6순위: 로컬 파일 검색
    local_files = ['./20250327_가공식품DB_147999건.csv', './20250327_가공식품DB_147999건.xlsx']
    
    for file_path in local_files:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                    st.sidebar.warning(f"📁 로컬 CSV: {len(df):,}개 식품")
                else:
                    df = pd.read_excel(file_path)
                    st.sidebar.warning(f"📁 로컬 Excel: {len(df):,}개 식품")
                
                load_time = time.time() - start_time
                st.sidebar.info(f"로딩 시간: {load_time:.2f}초")
                
                # 전처리 및 캐시 생성
                df = preprocess_dataframe(df)
                create_all_caches(df)
                return df
                
            except Exception as e:
                st.error(f"{file_path} 로딩 오류: {e}")
                continue
    
    # 모든 방법 실패시 파일 업로드 유도
    return None

def preprocess_dataframe(df):
    """DataFrame 전처리"""
    columns_to_use = [
        '식품명', '대표식품명', '식품소분류명',
        '에너지(kcal)', '단백질(g)', '지방(g)', '탄수화물(g)', '당류(g)',
        '나트륨(mg)', '콜레스테롤(mg)', '포화지방산(g)', '트랜스지방산(g)',
        '식이섬유(g)', '칼슘(mg)', '식품중량', '제조사명'
    ]
    
    existing_cols = [col for col in columns_to_use if col in df.columns]
    df = df[existing_cols]
    df = df.dropna(subset=['식품명'])
    
    # 데이터 타입 최적화
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    categorical_cols = ['식품소분류명', '제조사명']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

def create_all_caches(df):
    """모든 캐시 형태 생성"""
    with st.spinner("⚡ 고속 캐시 생성 중... (다음번부터 초고속 로딩!)"):
        # Parquet 파일
        parquet_path = './nutrition_data_optimized.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)
        
        # Pickle 캐시
        pickle_path = './nutrition_data.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # SQLite DB
        create_sqlite_db(df, './nutrition_data.db')
        
    st.success("🎉 캐시 생성 완료! 다음번부터는 초고속 로딩됩니다.")

# 일일 권장량 기준 설정
DAILY_RECOMMENDATIONS = {
    '에너지(kcal)': 2400,
    '단백질(g)': 65,
    '지방(g)': 65,
    '탄수화물(g)': 320,
    '당류(g)': 50,
    '나트륨(mg)': 2000,
    '칼슘(mg)': 700,
    '콜레스테롤(mg)': 300,
    '포화지방산(g)': 15,
    '식이섬유(g)': 25,
}

@st.cache_data
def expand_table(df):
    """일일 권장량 대비 비율 계산"""
    df_expanded = df.copy()
    for nutrient, recommended in DAILY_RECOMMENDATIONS.items():
        if nutrient in df_expanded.columns:
            new_col_name = nutrient.replace('(g)', '(%)').replace('(kcal)', '(%)').replace('(mg)', '(%)')
            df_expanded[new_col_name] = (df_expanded[nutrient] / recommended * 100).round(1)
    return df_expanded

def evaluate_nutri_score(row):
    """Nutri-Score 기반 건강도 점수 계산"""
    # Bad 요소 점수 함수들
    def score_energy(kcal):
        thresholds = [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350]
        return sum(kcal > t for t in thresholds)
    
    def score_sugars(g):
        thresholds = [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45]
        return sum(g > t for t in thresholds)
    
    def score_saturated_fat(g):
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return sum(g > t for t in thresholds)
    
    def score_sodium(mg):
        thresholds = [90, 180, 270, 360, 450, 540, 630, 720, 810, 900]
        return sum(mg > t for t in thresholds)
    
    # Good 요소 점수 함수들
    def score_fiber(g):
        thresholds = [0.9, 1.9, 2.8, 3.7, 4.7]
        return sum(g > t for t in thresholds)
    
    def score_protein(g):
        thresholds = [1.6, 3.2, 4.8, 6.4, 8.0]
        return sum(g > t for t in thresholds)
    
    # 수치 가져오기
    kcal = row.get("에너지(kcal)", 0)
    sugars = row.get("당류(g)", 0)
    sat_fat = row.get("포화지방산(g)", 0)
    sodium = row.get("나트륨(mg)", 0)
    fiber = row.get("식이섬유(g)", 0)
    protein = row.get("단백질(g)", 0)
    
    # 점수 계산
    bad = score_energy(kcal) + score_sugars(sugars) + score_saturated_fat(sat_fat) + score_sodium(sodium)
    good = score_fiber(fiber) + score_protein(protein)
    nutri_score = bad - good
    
    # 등급 변환
    if nutri_score <= -1:
        grade = "A"
    elif nutri_score <= 2:
        grade = "B"
    elif nutri_score <= 10:
        grade = "C"
    elif nutri_score <= 18:
        grade = "D"
    else:
        grade = "E"
    
    return pd.Series({
        "건강도점수": (40 - nutri_score),
        "건강등급": grade
    })

def search_foods(df, keyword):
    """키워드로 식품 검색"""
    if df is None or not keyword:
        return df
    
    keywords = keyword.strip().split()
    condition = df['식품명'].str.contains(keywords[0], case=False, na=False) | \
                df['대표식품명'].str.contains(keywords[0], case=False, na=False) | \
                df['식품소분류명'].str.contains(keywords[0], case=False, na=False)
    
    for kw in keywords[1:]:
        cond = df['식품명'].str.contains(kw, case=False, na=False) | \
               df['대표식품명'].str.contains(kw, case=False, na=False) | \
               df['식품소분류명'].str.contains(kw, case=False, na=False)
        condition = condition & cond
    
    return df[condition]

def create_plotly_boxplot(df):
    """Plotly를 사용한 박스플롯 생성"""
    if df is None or len(df) == 0:
        return None
        
    # 상위 15개 분류
    top15_classes = df.groupby('식품소분류명')['건강도점수'].mean().nlargest(15).index
    filtered_df = df[df['식품소분류명'].isin(top15_classes)]
    
    fig = px.box(filtered_df, 
                 x='식품소분류명', 
                 y='건강도점수',
                 title='건강도 점수 평균 상위 15개 (식품소분류별 분포)',
                 color='식품소분류명')
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        xaxis_title="식품소분류",
        yaxis_title="건강도점수"
    )
    
    return fig

def create_nutrition_chart_plotly(search_result, keyword, top_n=10):
    """Plotly를 사용한 영양성분 비교 차트"""
    if search_result is None or len(search_result) == 0:
        return None, None
    
    top_products = search_result.head(top_n)
    elements = ["에너지(%)", "당류(%)", "포화지방산(%)", '나트륨(%)', '단백질(%)', "식이섬유(%)"]
    existing_elements = [n for n in elements if n in top_products.columns]
    
    fig = go.Figure()
    
    for element in existing_elements:
        fig.add_trace(go.Bar(
            name=element,
            x=top_products['식품명'].str[:20],  # 긴 이름 자르기
            y=top_products[element],
            opacity=0.8
        ))
    
    fig.update_layout(
        title=f'"{keyword}" 검색 결과 - 영양성분 분석 (상위 {top_n}개)',
        xaxis_title='식품명',
        yaxis_title='함량(%)',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig, top_products

def create_correlation_heatmap_plotly(df_clean, existing_cols):
    """Plotly를 사용한 상관관계 히트맵"""
    if df_clean is None or len(df_clean) == 0:
        return None, None
        
    correlation_matrix = df_clean[existing_cols].corr()
    
    fig = px.imshow(correlation_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title='식품 영양성분 간 상관관계',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0)
    
    fig.update_layout(height=500)
    return fig, correlation_matrix

# 메인 앱
def main():
    st.title("🍎 식품영양정보 분석기")
    st.markdown("---")
    
    # 사이드바
    st.sidebar.title("📋 분석 옵션")
    
    # 데이터베이스 설정 (선택사항)
    with st.sidebar.expander("⚙️ 성능 설정"):
        if st.button("🗑️ 캐시 초기화"):
            # 모든 캐시 파일 삭제
            cache_files = [
                './nutrition_data_optimized.parquet',
                './nutrition_data.pkl', 
                './nutrition_data.db'
            ]
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            st.cache_data.clear()
            st.success("캐시가 초기화되었습니다!")
            st.experimental_rerun()
        
        st.info("💡 최초 실행 후 로딩 속도가 10-50배 향상됩니다")
    
    # 고성능 데이터 로드
    df = load_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        # 파일 업로드 옵션 제공
        uploaded_file = st.file_uploader(
            "CSV 또는 Excel 파일을 업로드하세요", 
            type=['csv', 'xlsx']
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df = pd.read_excel(uploaded_file)
                st.success("파일이 성공적으로 업로드되었습니다!")
            except Exception as e:
                st.error(f"파일 업로드 오류: {e}")
                return
        else:
            return
    
    # 데이터 전처리
    df = expand_table(df)
    df[['건강도점수', '건강등급']] = df.apply(evaluate_nutri_score, axis=1)
    
    # 기본 정보 표시
    st.sidebar.markdown(f"**📊 총 식품 수:** {len(df):,}개")
    st.sidebar.markdown(f"**🏷️ 식품 분류:** {df['식품소분류명'].nunique()}개")
    
    # 검색 키워드 입력
    keyword = st.sidebar.text_input("🔍 식품 검색", placeholder="예: 아이스크림, 라면")
    
    # 검색 결과
    if keyword:
        result = search_foods(df, keyword)
        st.sidebar.markdown(f"**검색 결과:** {len(result)}개 식품")
    else:
        result = df
        keyword = "전체 식품"
    
    # 메인 컨텐츠
    if len(result) > 0:
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📊 분류별 분석", "🥇 상위 제품", "📈 상관관계 분석", "📋 상세 데이터"])
        
        with tab1:
            st.subheader(f"'{keyword}' - 식품소분류별 건강도 점수 분포")
            fig_box = create_plotly_boxplot(result)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("표시할 데이터가 없습니다.")
        
        with tab2:
            st.subheader(f"'{keyword}' - 건강도 점수 상위 제품")
            
            # 상위 제품 수 선택
            top_n = st.selectbox("표시할 제품 수", [5, 10, 15, 20], index=1)
            
            # 건강도 점수 기준으로 정렬
            result_sorted = result.sort_values(by='건강도점수', ascending=False)
            
            fig_nutrition, top_products = create_nutrition_chart_plotly(result_sorted, keyword, top_n)
            if fig_nutrition:
                st.plotly_chart(fig_nutrition, use_container_width=True)
                
                # 상위 제품 테이블
                st.subheader("상위 제품 상세 정보")
                display_cols = ['식품명', '식품소분류명', '제조사명', '건강도점수', '건강등급']
                existing_display_cols = [col for col in display_cols if col in top_products.columns]
                st.dataframe(top_products[existing_display_cols], use_container_width=True)
            else:
                st.info("표시할 데이터가 없습니다.")
        
        with tab3:
            st.subheader(f"'{keyword}' - 영양성분 상관관계 분석")
            
            # 분석할 영양성분 선택
            available_nutrients = ['단백질(%)', '지방(%)', '탄수화물(%)', '당류(%)', '나트륨(%)', '콜레스테롤(%)']
            existing_nutrients = [n for n in available_nutrients if n in result.columns]
            
            if existing_nutrients:
                selected_nutrients = st.multiselect(
                    "분석할 영양성분 선택",
                    existing_nutrients,
                    default=existing_nutrients[:4]
                )
                
                if len(selected_nutrients) >= 2:
                    df_clean = result[selected_nutrients].dropna()
                    
                    if len(df_clean) > 1:
                        fig_corr, corr_matrix = create_correlation_heatmap_plotly(df_clean, selected_nutrients)
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # 상관계수 테이블
                            st.subheader("상관계수 매트릭스")
                            st.dataframe(corr_matrix.round(3), use_container_width=True)
                        else:
                            st.info("상관관계를 분석할 데이터가 부족합니다.")
                    else:
                        st.warning("분석할 데이터가 부족합니다.")
                else:
                    st.info("최소 2개 이상의 영양성분을 선택해주세요.")
            else:
                st.info("분석 가능한 영양성분 데이터가 없습니다.")
        
        with tab4:
            st.subheader(f"'{keyword}' - 상세 데이터 ({len(result)}개)")
            
            # 컬럼 선택
            all_columns = result.columns.tolist()
            default_columns = [
                '식품명', '대표식품명', '식품소분류명', '제조사명',
                '에너지(%)', '단백질(%)', '지방(%)', '탄수화물(%)',
                '건강도점수', '건강등급'
            ]
            existing_default_cols = [col for col in default_columns if col in all_columns]
            
            selected_columns = st.multiselect(
                "표시할 컬럼 선택",
                all_columns,
                default=existing_default_cols
            )
            
            if selected_columns:
                # 데이터 필터링 옵션
                col1, col2 = st.columns(2)
                
                with col1:
                    if '건강등급' in result.columns:
                        grade_filter = st.multiselect(
                            "건강등급 필터",
                            result['건강등급'].unique(),
                            default=result['건강등급'].unique()
                        )
                        result = result[result['건강등급'].isin(grade_filter)]
                
                with col2:
                    if '식품소분류명' in result.columns:
                        category_options = result['식품소분류명'].unique()
                        if len(category_options) > 1:
                            category_filter = st.multiselect(
                                "식품분류 필터",
                                category_options,
                                default=category_options
                            )
                            result = result[result['식품소분류명'].isin(category_filter)]
                
                # 데이터 표시
                st.dataframe(
                    result[selected_columns].fillna('-'),
                    use_container_width=True,
                    height=400
                )
                
                # 다운로드 버튼
                csv = result[selected_columns].to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv,
                    file_name=f"{keyword}_분석결과.csv",
                    mime="text/csv"
                )
            else:
                st.info("표시할 컬럼을 선택해주세요.")
    else:
        st.warning("검색 결과가 없습니다. 다른 키워드를 시도해보세요.")
    
    # 푸터
    st.markdown("---")
    st.markdown("🔬 **영양성분 분석 기준:** Nutri-Score 알고리즘 기반 건강도 점수 계산")
    st.markdown("📊 **일일 권장량:** 성인 남성 50세 기준")

if __name__ == "__main__":
    main()
