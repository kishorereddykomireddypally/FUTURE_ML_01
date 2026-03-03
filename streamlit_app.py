import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Sales Analytics & Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main {padding: 0rem 0rem;}
        [data-testid="stMetric"] {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
    # 📊 Sales Analytics & Forecasting Platform
    ### Intelligent insights powered by machine learning
    """)

# Load and preprocess data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datas.csv')
    df = pd.read_csv(data_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df['day'] = df['Order Date'].dt.day
    df['day_of_week'] = df['Order Date'].dt.dayofweek
    df['quarter'] = df['Order Date'].dt.quarter
    df['week'] = df['Order Date'].dt.isocalendar().week
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model(_df):
    # copy the dataframe to avoid mutating original
    df = _df.copy()
    X = df[['year', 'month', 'day', 'day_of_week', 'quarter']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2

model, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2 = train_model(df)

# Sidebar filters
st.sidebar.markdown("### 🔧 Dashboard Settings")
date_range = st.sidebar.date_input(
    "Select date range:",
    value=(df['Order Date'].min().date(), df['Order Date'].max().date()),
    min_value=df['Order Date'].min().date(),
    max_value=df['Order Date'].max().date()
)

filtered_df = df[(df['Order Date'].dt.date >= date_range[0]) & 
                 (df['Order Date'].dt.date <= date_range[1])]

forecast_days = st.sidebar.slider(
    "Forecast period (days):",
    min_value=7,
    max_value=90,
    value=30,
    step=7
)

# Key metrics
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Sales",
        value=f"${filtered_df['Sales'].sum():,.0f}",
        delta=f"${filtered_df['Sales'].sum() - df['Sales'].mean():,.0f}" if len(filtered_df) > 0 else None,
    )

with col2:
    st.metric(
        label="Avg Sales",
        value=f"${filtered_df['Sales'].mean():,.0f}",
        delta=f"{((filtered_df['Sales'].mean() / df['Sales'].mean() - 1) * 100):.1f}%" if df['Sales'].mean() > 0 else None,
    )

with col3:
    st.metric(
        label="Total Orders",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df) // 10:,}" if len(filtered_df) > 0 else None,
    )

with col4:
    st.metric(
        label="Model R²",
        value=f"{r2:.3f}",
        delta="Performance Score"
    )

with col5:
    st.metric(
        label="Forecast RMSE",
        value=f"${rmse:,.0f}",
        delta="Error Margin"
    )

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Analytics", "🔮 Forecast", "📊 Data Explorer", "⚙️ Model Performance"]
)

# Tab 1: Analytics
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Sales Trend")
        daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
        st.line_chart(daily_sales.set_index('Order Date')['Sales'])
    
    with col2:
        st.subheader("Sales by Category")
        category_sales = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        category_sales.plot(kind='barh', ax=ax, color='#667eea')
        ax.set_xlabel('Total Sales ($)')
        st.pyplot(fig, width='stretch')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Monthly Sales")
        monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum()
        st.bar_chart(monthly_sales)
    
    with col4:
        st.subheader("Top 10 Products")
        top_products = filtered_df.groupby('Product Name')['Sales'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        top_products.plot(kind='barh', ax=ax, color='#764ba2')
        ax.set_xlabel('Sales ($)')
        st.pyplot(fig, width='stretch')

# Tab 2: Forecast
with tab2:
    st.subheader(f"Sales Forecast - Next {forecast_days} Days")
    
    future_dates = pd.date_range(
        start=df['Order Date'].max() + pd.Timedelta(days=1),
        periods=forecast_days
    )
    future_df = pd.DataFrame({'Order Date': future_dates})
    future_df['year'] = future_df['Order Date'].dt.year
    future_df['month'] = future_df['Order Date'].dt.month
    future_df['day'] = future_df['Order Date'].dt.day
    future_df['day_of_week'] = future_df['Order Date'].dt.dayofweek
    future_df['quarter'] = future_df['Order Date'].dt.quarter
    
    future_sales = model.predict(
        future_df[['year', 'month', 'day', 'day_of_week', 'quarter']]
    )
    future_df['forecast_sales'] = future_sales
    
    # Ensure no negative forecasts
    future_df['forecast_sales'] = future_df['forecast_sales'].clip(lower=0)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot historical data
        historical = df.groupby('Order Date')['Sales'].sum().tail(60)
        ax.plot(historical.index, historical.values, label='Historical', color='#667eea', linewidth=2)
        
        # Plot forecast
        ax.plot(future_df['Order Date'], future_df['forecast_sales'], 
               label='Forecast', color='#764ba2', linestyle='--', linewidth=2)
        
        ax.fill_between(future_df['Order Date'], 
                         future_df['forecast_sales'] * 0.85,
                         future_df['forecast_sales'] * 1.15,
                         alpha=0.2, color='#764ba2', label='Confidence Interval (±15%)')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales ($)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, width='stretch')
    
    with col2:
        st.metric("Total Forecast", f"${future_df['forecast_sales'].sum():,.0f}")
        st.metric("Average Daily Forecast", f"${future_df['forecast_sales'].mean():,.0f}")
        st.metric("Max Forecast Day", f"${future_df['forecast_sales'].max():,.0f}")
        st.metric("Min Forecast Day", f"${future_df['forecast_sales'].min():,.0f}")
    
    # Forecast table
    with st.expander("📋 Detailed Forecast Table"):
        forecast_display = future_df[['Order Date', 'forecast_sales']].copy()
        forecast_display.columns = ['Date', 'Forecasted Sales ($)']
        forecast_display['Forecasted Sales ($)'] = forecast_display['Forecasted Sales ($)'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(forecast_display, width='stretch', hide_index=True)

# Tab 3: Data Explorer
with tab3:
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{(df['Order Date'].max() - df['Order Date'].min()).days} days")
    with col3:
        st.metric("Unique Customers", f"{df['Customer ID'].nunique():,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Sales'], bins=50, color='#667eea', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Sales ($)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig, width='stretch')
    
    with col2:
        st.subheader("Segment Distribution")
        segment_counts = df['Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#667eea', '#764ba2', '#f093fb']
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        st.pyplot(fig, width='stretch')
    
    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(df.sort_values('Order Date', ascending=False).head(100), width='stretch')

# Tab 4: Model Performance
with tab4:
    st.subheader("Model Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}", "Lower is better")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}", "Lower is better")
    with col3:
        st.metric("R² Score", f"{r2:.4f}", "Closer to 1 is better")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted (Test Set)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, alpha=0.5, color='#667eea')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Sales ($)')
        ax.set_ylabel('Predicted Sales ($)')
        ax.legend()
        st.pyplot(fig, width='stretch')
    
    with col2:
        st.subheader("Residuals Plot")
        residuals = y_test.values - y_pred
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_pred, residuals, alpha=0.5, color='#764ba2')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Sales ($)')
        ax.set_ylabel('Residuals ($)')
        st.pyplot(fig, width='stretch')
    
    st.markdown("---")
    st.subheader("Feature Importance")
    features = ['year', 'month', 'day', 'day_of_week', 'quarter']
    importance = np.abs(model.coef_)
    importance = importance / importance.sum() * 100
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#667eea' if i != np.argmax(importance) else '#764ba2' for i in range(len(features))]
    ax.barh(features, importance, color=colors)
    ax.set_xlabel('Relative Importance (%)')
    st.pyplot(fig, width='stretch')

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em; padding: 20px;'>
        <p>💼 Sales Analytics Platform | Powered by Machine Learning</p>
        <p>Last Updated: """ + df['Order Date'].max().strftime('%Y-%m-%d') + """ | Data Points: """ + f"{len(df):,}" + """</p>
    </div>
    """, unsafe_allow_html=True)
