import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
# Gowun Dodum 구글 폰트 적용 및 전반적인 CSS 스타일 지정
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Gowun Dodum', sans-serif !important;
    }

    .main-header {
        font-family: 'Gowun Dodum', sans-serif !important;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Gowun Dodum', sans-serif !important;
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        font-family: 'Gowun Dodum', sans-serif !important;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-card-secondary {
        font-family: 'Gowun Dodum', sans-serif !important;
        background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-family: 'Gowun Dodum', sans-serif !important;
        font-size: 3rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    .metric-label {
        font-family: 'Gowun Dodum', sans-serif !important;
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.95;
    }
    .info-box {
        font-family: 'Gowun Dodum', sans-serif !important;
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8')
    except Exception as e:
        df = pd.read_csv(url, encoding='cp949')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# Ensure date column is datetime
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
df['연도'] = df['날짜'].dt.year

# Filter criteria: up to 2025
df_filtered_period = df[(df['연도'] >= 1900) & (df['연도'] <= 2025)]

# Calculate count of days and annual average temperature
annual_stats = df_filtered_period.groupby('연도').agg(
    일수=('날짜', 'count'),
    평균기온=('평균기온', 'mean')
).reset_index()

# Filter out years with less than 300 observation days
valid_annual = annual_stats[annual_stats['일수'] >= 300].dropna(subset=['평균기온'])

# --- 전체 기간 회귀 모델 ---
X = valid_annual[['연도']]
y = valid_annual['평균기온']

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r_value = np.corrcoef(valid_annual['연도'], valid_annual['평균기온'])[0, 1]

# 100년당 기온 상승폭 계산 (기울기 * 100)
slope_per_century = slope * 100

# --- 최근 20년 회귀 모델 (예: 2006년 ~ 2025년) ---
recent_valid = valid_annual[valid_annual['연도'] >= (2025 - 20, 2025)[1] - 20] # 최근 20년 (2006~2025)
X_recent = recent_valid[['연도']]
y_recent = recent_valid['평균기온']

model_recent = LinearRegression()
model_recent.fit(X_recent, y_recent)

slope_recent = model_recent.coef_[0]
slope_recent_per_century = slope_recent * 100

# Model summary stats
num_years = len(valid_annual)
start_year = int(valid_annual['연도'].min())
end_year = int(valid_annual['연도'].max())

st.markdown('<p class="main-header">🌡️ 서울 기온 예측기 (1900~2100)</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">서울 기온 데이터를 활용하여 연도별 평균기온 추이를 분석하고 미래 기온을 예측합니다. (기준 기간: ~2025년, 300일 미만 관측 연도 제외)</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎛️ 예측 설정")
selected_year = st.sidebar.slider(
    "조회/예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

st.markdown("### 📊 분석 데이터 요약")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="분석에 사용된 연도 수", value=f"{num_years}년")
with m2:
    st.metric(label="시작 연도", value=f"{start_year}년")
with m3:
    st.metric(label="끝 연도", value=f"{end_year}년")
with m4:
    st.metric(label="상관계수 (r)", value=f"{r_value:.3f}")

st.markdown("---")

# --- 100년당 기온 상승폭 비교 카드 섹션 ---
st.markdown("### 📈 100년당 기온 상승 속도 비교")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">전체 기간 ({start_year}~{end_year}) 기준</div>
            <div class="metric-value">+{slope_per_century:.2f} ℃ / 100년</div>
            <div style="margin-top: 0.3rem; font-size: 1.1rem; opacity: 0.9;">장기적인 기온 상승 추세</div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);">
            <div class="metric-label">최근 20년 (2006~2025) 기준</div>
            <div class="metric-value">+{slope_recent_per_century:.2f} ℃ / 100년</div>
            <div style="margin-top: 0.3rem; font-size: 1.1rem; opacity: 0.9;">최근 가속화된 온난화 속도</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

predicted_temp = model.predict([[selected_year]])[0]

# Check if selected year exists in historical filtered data
historical_match = valid_annual[valid_annual['연도'] == selected_year]
is_historical = not historical_match.empty

st.markdown(f"### 🎯 **{selected_year}년** 기온 정보")

# Big prominent display card for the selected year temperature
if is_historical:
    actual_temp = historical_match['평균기온'].values[0]
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #10B981 0%, #047857 100%);">
            <div class="metric-label">{selected_year}년 실제 관측 연평균기온</div>
            <div class="metric-value">{actual_temp:.2f} ℃</div>
            <div style="margin-top: 0.3rem; font-size: 1.1rem; opacity: 0.9;">회귀 직선 예측값: {predicted_temp:.2f} ℃</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #10B981 0%, #047857 100%);">
            <div class="metric-label">{selected_year}년 회귀 분석 예상 연평균기온</div>
            <div class="metric-value">{predicted_temp:.2f} ℃</div>
            <div style="margin-top: 0.3rem; font-size: 1.1rem; opacity: 0.9;">전체 기간 100년당 상승률 적용: +{slope_per_century:.2f} ℃ / 100년</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 📈 서울 연평균기온 산점도 및 회귀 직선")

# Generate line points for regression line across the full range
reg_x = np.arange(1900, 2101)
reg_y = model.predict(reg_x.reshape(-1, 1))

fig = go.Figure()

# Scatter plot of historical valid annual temperatures
fig.add_trace(go.Scatter(
    x=valid_annual['연도'],
    y=valid_annual['평균기온'],
    mode='markers',
    name='연평균기온 (실제)',
    marker=dict(color='#3B82F6', size=8, opacity=0.8)
))

# Regression line
fig.add_trace(go.Scatter(
    x=reg_x,
    y=reg_y,
    mode='lines',
    name='회귀 직선 (Trend)',
    line=dict(color='#EF4444', width=3)
))

# Highlight marker for selected year
if selected_year >= 1900 and selected_year <= 2100:
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode='markers',
        name=f'선택 연도 ({selected_year}년)',
        marker=dict(color='#10B981', size=16, symbol='star')
    ))

fig.update_layout(
    title=f"서울 연도별 평균기온 추이 및 회귀 분석 (전체 vs 최근 20년)",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    hovermode="x unified",
    template="plotly_white",
    height=550,
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(
        family="Gowun Dodum, sans-serif"
    )
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ 데이터 처리 기준 및 분석 설명 보기"):
    st.markdown(f"""
    - **기준 기간**: 2025년까지의 데이터를 대상으로 합니다.
    - **데이터 필터링**: 관측 일수가 300일 미만인 부실 데이터 연도는 정확한 연평균 산출을 위해 제외되었습니다.
    - **총 분석 대상 연도 수**: {num_years}개년 ({start_year}년 ~ {end_year}년)
    - **상관계수 (r)**: {r_value:.4f} (연도와 평균기온 간의 선형 상관관계 정도를 나타냅니다.)
    - **100년당 기온 상승폭 (전체)**: +{slope_per_century:.2f} ℃ / 100년
    - **100년당 기온 상승폭 (최근 20년)**: +{slope_recent_per_century:.2f} ℃ / 100년
    - **회귀 방정식**: $y = {slope:.4f} \\times \\text{{연도}} + {intercept:.4f}$
    """)
