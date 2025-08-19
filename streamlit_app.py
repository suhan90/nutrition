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

# 캐시된 데이터 로딩 함수
@st.cache_data
def load_data():
    """데이터를 로드하고 전처리"""
    # 파일 업로드 또는 기본 파일 사용
    csv_file_path = './20250327_가공식품DB_147999건.csv'
    xls_file_path = './20250327_가공식품DB_147999건.xlsx'
    
    try:
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False)
        elif os.path.exists(xls_file_path):
            df = pd.read_excel(xls_file_path)
        else:
            st.error("데이터 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
            return None
            
        # 사용할 컬럼 선택
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
        
        return df
        
    except Exception as e:
        st.error(f"데이터 로딩 오류: {e}")
        return None

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
    
    # 데이터 로드
    with st.spinner("데이터를 로딩중입니다..."):
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
