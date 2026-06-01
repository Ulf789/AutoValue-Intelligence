# app.py
import streamlit as st
from model_engine import train_and_predict
from ai_agent import generate_market_report

# 1. Basic page configuration (set page title and wide layout)
st.set_page_config(page_title="AutoValue Intelligence", page_icon="🚗", layout="wide")

# 2. Main page header
st.title("🚗 AutoValue Intelligence: AI-Powered Market & Valuation Assistant")
st.markdown("---")

# 3. Sidebar configuration: let the buyer enter vehicle details
st.sidebar.header("🛠️ Enter Vehicle Specifications")

# Vehicle model dropdown
model_name = st.sidebar.selectbox(
    "Select Vehicle Model", 
    ["BMW M5", "3 Series Touring", "Audi R8"]
)

# First registration year slider (2016 - 2026)
year = st.sidebar.slider("First Registration Year", 2016, 2026, 2021)

# Mileage input box (step size: 5,000 km)
mileage = st.sidebar.number_input(
    "Current Mileage (in km)", 
    min_value=1000, 
    max_value=300000, 
    value=60000, 
    step=5000
)

# Seller's asking price input box (step size: €1,000)
user_price = st.sidebar.number_input(
    "Seller's Asking Price (EUR)", 
    min_value=5000, 
    max_value=300000, 
    value=89000, 
    step=1000
)

# 4. Core trigger button
if st.sidebar.button("Run Market Intelligence Analysis", type="primary"):
    
    # Loading spinner animations shown on the page
    with st.spinner("Step 1: Training Random Forest Regression Engine on Market Data..."):
        # Call the ML engine to compute the system's estimated fair value
        predicted_price = train_and_predict(model_name, year, mileage)
        
    with st.spinner("Step 2: Synthesizing Technical Vulnerabilities via LangChain Agent..."):
        # Call the LangChain engine to generate the full English expert report
        report = generate_market_report(model_name, year, mileage, user_price, predicted_price)
        
    # 5. Display quantitative results (metric cards)
    st.subheader("📊 Market Pricing Overview")
    col1, col2, col3 = st.columns(3)
    
    # Show seller's asking price
    col1.metric("Seller Asking Price", f"€{user_price:,}")
    # Show ML-predicted fair baseline price
    col2.metric("AI Predicted Base Price", f"€{predicted_price:,}")
    
    # Calculate the price difference
    price_diff = predicted_price - user_price
    if price_diff > 0:
        # If the predicted price is higher than the asking price, it's a good deal
        col3.metric("Market Deal Status", "Good Deal", delta=f"+€{abs(price_diff):,} (Underpriced)")
    else:
        # If the predicted price is lower than the asking price, the seller is charging a premium
        col3.metric("Market Deal Status", "Overpriced", delta=f"-€{abs(price_diff):,} (Premium Required)")
        
    st.markdown("---")
    
    # 6. Display qualitative results (render the model's Markdown report directly)
    st.subheader("📋 AI Expert Deep Analysis & Deal Strategy")
    st.markdown(report)
