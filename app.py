# app.py
import streamlit as st
from model_engine import train_and_predict
from ai_agent import generate_market_report

# 1. 网页基础配置（设置网页标题和宽屏布局）
st.set_page_config(page_title="AutoValue Intelligence", page_icon="🚗", layout="wide")

# 2. 网页大标题
st.title("🚗 AutoValue Intelligence: AI-Powered Market & Valuation Assistant")
st.markdown("---")

# 3. 侧边栏配置：让买家输入车辆信息
st.sidebar.header("🛠️ Enter Vehicle Specifications")

# 车型选择下拉框
model_name = st.sidebar.selectbox(
    "Select Vehicle Model", 
    ["BMW M5", "3 Series Touring", "Audi R8"]
)

# 首次上牌年份滑块（2016 - 2026）
year = st.sidebar.slider("First Registration Year", 2016, 2026, 2021)

# 里程数输入框（步长 5000 km）
mileage = st.sidebar.number_input(
    "Current Mileage (in km)", 
    min_value=1000, 
    max_value=300000, 
    value=60000, 
    step=5000
)

# 卖家开价输入框（步长 1000 欧）
user_price = st.sidebar.number_input(
    "Seller's Asking Price (EUR)", 
    min_value=5000, 
    max_value=300000, 
    value=89000, 
    step=1000
)

# 4. 核心触发按钮
if st.sidebar.button("Run Market Intelligence Analysis", type="primary"):
    
    # 网页上的加载动画效果
    with st.spinner("Step 1: Training Random Forest Regression Engine on Market Data..."):
        # 调用第二步的机器学习引擎计算系统估值
        predicted_price = train_and_predict(model_name, year, mileage)
        
    with st.spinner("Step 2: Synthesizing Technical Vulnerabilities via LangChain Agent..."):
        # 调用第三步的 LangChain 引擎生成全英文专家报告
        report = generate_market_report(model_name, year, mileage, user_price, predicted_price)
        
    # 5. 展示定量分析结果（炫酷的数字卡片）
    st.subheader("📊 Market Pricing Overview")
    col1, col2, col3 = st.columns(3)
    
    # 显示卖家要价
    col1.metric("Seller Asking Price", f"€{user_price:,}")
    # 显示机器学习预测的合理基准价
    col2.metric("AI Predicted Base Price", f"€{predicted_price:,}")
    
    # 计算差价
    price_diff = predicted_price - user_price
    if price_diff > 0:
        # 如果系统预测价高于卖家开价，说明捡漏了
        col3.metric("Market Deal Status", "Good Deal", delta=f"+€{abs(price_diff):,} (Underpriced)")
    else:
        # 如果系统预测价低于卖家开价，说明买贵了
        col3.metric("Market Deal Status", "Overpriced", delta=f"-€{abs(price_diff):,} (Premium Required)")
        
    st.markdown("---")
    
    # 6. 展示定性分析结果（直接渲染大模型的 Markdown 英文报告）
    st.subheader("📋 AI Expert Deep Analysis & Deal Strategy")
    st.markdown(report)