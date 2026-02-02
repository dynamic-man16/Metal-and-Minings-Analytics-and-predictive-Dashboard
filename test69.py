import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import requests

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------

# Ideally, put this in st.secrets, but keeping here for your ease of running
NEWS_API_KEY = "d3d096e3894b496b8302a4e555c1f105"

COMPANY_NAMES = {
    "TATASTEEL.NS": "Tata Steel",
    "SAIL.NS": "Steel Authority of India",
    "HINDALCO.NS": "Hindalco Industries",
    "NMDC.NS": "NMDC Limited",
    "MOIL.NS": "MOIL Limited",
    "JINDALSAW.NS": "Jindal SAW"
}

# Static Data with added ROE fallbacks
STATIC_DATA = {
    "TATASTEEL.NS": {
        "trend_txt": "Revenue Consolidation: -5.9% dip in 2025 vs 2024.",
        "comp_txt": "EBITDA Margin: Outperforming Sector Average (14% vs 11%).",
        "last_price": 145.00,
        "roe": 0.075 # 7.5% fallback
    },
    "SAIL.NS": {
        "trend_txt": "Stable Revenue > ₹1L Cr. Slight dip in 2025 (-2.7%).",
        "comp_txt": "Liquidity: Strongest Cash Flow in Public Sector Steel.",
        "last_price": 95.00,
        "roe": 0.042 # 4.2% fallback
    },
    "HINDALCO.NS": {
        "trend_txt": "Strong Growth: +12% Revenue Jump in 2025.",
        "comp_txt": "Growth Leader: Highest Top-line growth among peers.",
        "last_price": 520.00,
        "roe": 0.115 # 11.5% fallback
    },
    "NMDC.NS": {
        "trend_txt": "Robust Expansion: +11% Revenue Growth YoY.",
        "comp_txt": "Valuation: Lowest P/E (9.5X) suggests deep value.",
        "last_price": 235.00,
        "roe": 0.260 # 26.0% fallback
    },
    "MOIL.NS": {
        "trend_txt": "Steady Incline: Consistent ~9% YoY Growth.",
        "comp_txt": "Dividends: Highest yield potential in small-cap mining.",
        "last_price": 310.00,
        "roe": 0.185 # 18.5% fallback
    },
    "JINDALSAW.NS": {
        "trend_txt": "Plateauing Revenue: Flat growth in 2025 vs 2024.",
        "comp_txt": "Profitability: Highest EPS (₹29.44) in peer group.",
        "last_price": 415.00,
        "roe": 0.148 # 14.8% fallback
    }
}

# Analysis Data - 10 Year Trends
ANALYSIS_DATA = {
    "TATASTEEL.NS": {
        "trend": pd.DataFrame({
            "Metric": ["Revenue", "Expenses", "EBIT", "Reserves & Surplus", "Net Fixed Assets"],
            "2016": [100, 100, 100, 100, 100], 
            "2017": [124.7, 117.2, 186.6, 106.6, 144.8], 
            "2018": [141.7, 127.4, 259.3, 132.1, 143.2], 
            "2019": [165.4, 141.5, 360.5, 151.7, 142.1], 
            "2020": [141.5, 130.1, 235.4, 160.7, 142.2],
            "2021": [197.0, 164.2, 465.8, 204.0, 193.9], 
            "2022": [302.2, 218.6, 986.2, 271.9, 188.2], 
            "2023": [334.7, 316.9, 480.0, 296.4, 194.5], 
            "2024": [330.1, 306.6, 522.3, 306.5, 196.4], 
            "2025": [310.4, 291.5, 464.9, 274.7, 198.6]
        })
    },
    "SAIL.NS": {
        "trend": pd.DataFrame({
            "Metric": ["Revenue", "Expenses", "EBIT", "Reserves & Surplus", "Net Fixed Assets"],
            "2016": [100, 100, 100, 100, 100], 
            "2017": [113.4, 107.9, -56.1, 90.9, 109.5], 
            "2018": [134.4, 117.2, 42.8, 90.0, 127.6], 
            "2019": [152.6, 124.8, 134.8, 97.0, 133.6], 
            "2020": [140.5, 113.7, 136.8, 101.6, 150.3],
            "2021": [157.5, 124.5, 183.2, 112.2, 147.2], 
            "2022": [235.8, 177.9, 362.0, 136.5, 160.4], 
            "2023": [238.1, 208.7, 65.0, 136.9, 160.1], 
            "2024": [240.2, 204.8, 124.3, 142.5, 157.7], 
            "2025": [233.6, 200.7, 105.7, 146.9, 159.5]
        })
    },
    "HINDALCO.NS": {
        "trend": pd.DataFrame({
            "Metric": ["Revenue", "Expenses", "EBIT", "Reserves & Surplus", "Long-Term Investments"],
            "2016": [100, 100, 100, 100, 100], 
            "2017": [107.3, 103.6, 143.9, 112.2, 103.8], 
            "2018": [118.3, 114.8, 153.1, 117.3, 117.8], 
            "2019": [124.6, 124.4, 126.9, 115.2, 110.1], 
            "2020": [109.6, 109.6, 109.5, 107.9, 100.3],
            "2021": [116.3, 115.7, 122.4, 118.7, 123.5], 
            "2022": [184.3, 169.2, 334.7, 129.2, 127.2], 
            "2023": [209.4, 208.2, 221.8, 138.8, 121.6], 
            "2024": [226.1, 226.3, 224.1, 151.3, 135.9], 
            "2025": [254.2, 246.2, 333.6, 166.8, 141.6]
        })
    },
    "NMDC.NS": {
        "trend": pd.DataFrame({
            "Metric": ["Revenue", "Expenses", "EBIT", "Reserves & Surplus", "Net Fixed Assets"],
            "2016": [100, 100, 100, 100, 100], 
            "2017": [136.7, 196.4, 103.8, 77.3, 101.7], 
            "2018": [179.9, 234.8, 149.5, 83.6, 139.2], 
            "2019": [188.2, 213.7, 174.1, 89.2, 141.8], 
            "2020": [181.2, 242.1, 147.5, 94.8, 157.2],
            "2021": [238.0, 280.7, 214.5, 102.5, 162.5], 
            "2022": [402.1, 561.2, 314.1, 61.2, 147.7], 
            "2023": [273.6, 433.0, 185.5, 76.7, 157.0], 
            "2024": [329.8, 574.3, 194.6, 87.4, 164.8], 
            "2025": [366.5, 617.4, 227.9, 99.9, 250.3]
        })
    },
    "MOIL.NS": {
        "trend": pd.DataFrame({
            "Metric": ["Revenue", "Expenses", "EBIT", "Reserves & Surplus", "Equity Capital"],
            "2016": [100, 100, 100, 100, 100], 
            "2017": [156.0, 121.5, 1330.2, 81.3, 79.2], 
            "2018": [208.6, 138.4, 2597.8, 77.3, 153.3], 
            "2019": [227.0, 147.9, 2922.3, 85.9, 153.3], 
            "2020": [163.6, 142.5, 880.6, 76.8, 141.2],
            "2021": [185.5, 160.5, 1036.7, 78.6, 141.2], 
            "2022": [226.3, 162.0, 2417.7, 58.9, 121.1], 
            "2023": [211.4, 176.3, 1407.5, 62.1, 121.1], 
            "2024": [228.4, 187.5, 1621.3, 68.4, 121.1], 
            "2025": [249.8, 196.2, 2074.1, 74.0, 121.1]
        })
    },
    "JINDALSAW.NS": {
        "trend": pd.DataFrame({
            "Metric": ["Revenue", "Expenses", "EBIT", "Reserves & Surplus", "Net Fixed Assets"],
            "2016": [100, 100, 100, 100, 100], 
            "2017": [93.4, 84.6, 150.3, 107.4, 100.2], 
            "2018": [116.2, 115.9, 150.6, 114.2, 97.9], 
            "2019": [155.1, 155.1, 189.1, 123.1, 98.9], 
            "2020": [159.8, 163.1, 185.9, 132.8, 103.7],
            "2021": [136.2, 143.7, 129.7, 138.2, 102.7], 
            "2022": [174.0, 170.3, 164.9, 145.0, 101.6], 
            "2023": [241.2, 243.3, 240.0, 157.0, 103.4], 
            "2024": [283.5, 271.1, 560.0, 197.9, 127.4], 
            "2025": [283.1, 260.3, 636.7, 232.0, 135.4]
        })
    }
}

st.set_page_config(
    page_title="Metals & Mining Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="⛏️"
)

# IMPROVED AESTHETIC CSS - TradingView Style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp { background-color: #0b0b0b; }
    #MainMenu, footer, header { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none; }
    
    /* Card System */
    .card {
        background: linear-gradient(145deg, #131313, #0f0f0f);
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    
    .card-header {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #666;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1a1a1a;
    }
    
    /* Control Bar */
    .control-bar {
        background: linear-gradient(135deg, #131313, #0f0f0f);
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Metrics */
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 16px 0;
    }
    
    .metric-box {
        text-align: center;
        padding: 12px;
        background-color: #0d0d0d;
        border-radius: 6px;
        border: 1px solid #1a1a1a;
        flex: 1;
        margin: 0 6px;
    }
    
    .metric-val {
        font-size: 20px;
        font-weight: 700;
        color: #fff;
        margin-bottom: 4px;
    }
    
    .metric-lbl {
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
    }
    
    /* Ticker Tape */
    .ticker-wrap {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 36px;
        background-color: #080808;
        border-top: 1px solid #1a1a1a;
        z-index: 999999;
        overflow: hidden;
        display: flex;
        align-items: center;
    }
    
    .ticker-content {
        display: inline-block;
        font-size: 13px;
        font-weight: 500;
        color: #999;
        white-space: nowrap;
        padding-left: 100%;
        animation: ticker-scroll 60s linear infinite;
    }
    
    @keyframes ticker-scroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    
    /* Streamlit Widget Styling */
    .stSelectbox > div > div {
        background-color: #0d0d0d !important;
        border: 1px solid #262626 !important;
        color: #fff !important;
    }
    
    .stRadio > div { background-color: transparent !important; }
    
    .stRadio label {
        color: #999 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    
    .stSlider > div > div { background-color: #0d0d0d !important; }
    .stSlider span { color: #fff !important; }
    
    .stButton > button {
        background-color: #1a1a1a;
        color: #26a69a;
        border: 1px solid #262626;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #262626;
        border-color: #26a69a;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d0d0d;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #131313;
        border: 1px solid #1a1a1a;
        color: #999;
        border-radius: 6px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a;
        color: #26a69a;
        border-color: #26a69a;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def calculate_smart_roe(info, ticker):
    """
    Robustly calculates ROE using three layers of fallback:
    1. Direct API fetch (info['returnOnEquity'])
    2. Manual Calculation (Net Income / Total Equity)
    3. Static Data Fallback (Hardcoded values)
    """
    # 1. Try Direct Fetch
    roe = info.get('returnOnEquity')
    
    # 2. Try Manual Calculation if Direct Fetch returns None
    if roe is None:
        try:
            # Net Income
            net_income = info.get('netIncomeToCommon')
            # Total Equity
            equity = info.get('totalStockholderEquity')
            
            # If those are missing, try getting book value * shares
            if equity is None:
                book_val = info.get('bookValue')
                shares = info.get('sharesOutstanding')
                if book_val and shares:
                    equity = book_val * shares
            
            if net_income and equity and equity != 0:
                roe = net_income / equity
        except Exception:
            pass
            
    # 3. Fallback to Static Data if still None
    if roe is None:
        roe = STATIC_DATA.get(ticker, {}).get("roe", 0)
        
    return roe

@st.cache_data(ttl=300)
def fetch_company_news(symbol, company_name):
    """Fetch company-specific news from NewsAPI"""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{company_name}" OR {symbol.replace(".NS", "")}',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") != "ok":
            return []
        
        headlines = []
        for article in data.get("articles", []):
            title = article.get("title", "")
            description = article.get("description", "") or ""
            combined = f"{title} {description}".lower()
            
            company_lower = company_name.lower()
            symbol_clean = symbol.replace(".NS", "").lower()
            
            if company_lower in combined or symbol_clean in combined:
                headlines.append(title)
                
            if len(headlines) >= 3:
                break
        
        return headlines if headlines else ["No recent news"]
        
    except Exception:
        return ["News unavailable"]


@st.cache_data(ttl=300)
def fetch_live_data(ticker):
    """Fetch live market data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y", interval="1d")
        
        if hist.empty:
            hist = stock.history(period="max", interval="1d")
            
        if hist.empty:
            dates = pd.date_range(end=datetime.now(), periods=100)
            dummy_price = STATIC_DATA.get(ticker, {}).get("last_price", 100.0)
            hist = pd.DataFrame({
                'Open': [dummy_price]*100, 'High': [dummy_price]*100, 
                'Low': [dummy_price]*100, 'Close': [dummy_price]*100, 
                'Volume': [0]*100
            }, index=dates)
            current_price = dummy_price
            return hist, current_price, {}

        try:
            todays_data = stock.history(period="1d", interval="1m")
            if not todays_data.empty:
                current_price = todays_data['Close'].iloc[-1]
            else:
                current_price = hist['Close'].iloc[-1]
        except:
            current_price = hist['Close'].iloc[-1]
            
        hist.reset_index(inplace=True)
        if 'Date' not in hist.columns and 'Datetime' in hist.columns:
             hist.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'Date' not in hist.columns:
             hist['Date'] = hist.index

        return hist, current_price, stock.info
        
    except Exception as e:
        dates = pd.date_range(end=datetime.now(), periods=100)
        dummy_price = STATIC_DATA.get(ticker, {}).get("last_price", 100.0)
        hist = pd.DataFrame({
            'Date': dates, 
            'Open': [dummy_price]*100, 'High': [dummy_price]*100, 
            'Low': [dummy_price]*100, 'Close': [dummy_price]*100, 
            'Volume': [0]*100
        })
        return hist, dummy_price, {}


# ML Functions
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_analytics(df, days_forecast):
    if len(df) < 50 or df['Close'].std() == 0:
        df = df.copy()
        df['Close'] = df['Close'].ffill() # Updated deprecated method
        df['Close'] += np.random.normal(0, 0.001, len(df))

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())
    df['MA_50'] = df['Close'].rolling(50).mean().bfill()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close']).fillna(50)
    df['Lag_1'] = df['Close'].shift(1).bfill()
    df['Lag_2'] = df['Close'].shift(2).bfill()

    features = ['Date_Ordinal', 'MA_50', 'EMA_20', 'RSI', 'Lag_1', 'Lag_2']
    X = df[features]
    y = df['Close']

    split = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_dates = df['Date'].iloc[split:]

    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    nn = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    )

    rf.fit(X_train, y_train)
    nn.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    nn_pred = nn.predict(X_test)

    metrics = {
        "RF_RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "NN_RMSE": np.sqrt(mean_squared_error(y_test, nn_pred)),
        "RF_MAPE": np.mean(np.abs((y_test - rf_pred) / y_test)) * 100,
        "NN_MAPE": np.mean(np.abs((y_test - nn_pred) / y_test)) * 100
    }

    last = df.iloc[-1]
    future_ord = last['Date_Ordinal'] + days_forecast

    future_X = pd.DataFrame([{
        'Date_Ordinal': future_ord,
        'MA_50': last['MA_50'],
        'EMA_20': last['EMA_20'],
        'RSI': last['RSI'],
        'Lag_1': last['Close'],
        'Lag_2': last['Lag_1']
    }])

    future_price = (rf.predict(future_X)[0] + nn.predict(future_X)[0]) / 2

    return {
        "dates": test_dates,
        "actual": y_test.values,
        "rf_pred": rf_pred,
        "nn_pred": nn_pred,
        "future_price": future_price,
        "metrics": metrics
    }


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
st.markdown('<div class="control-bar">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.4, 2.8, 1.6])

with c1:
    category = st.radio("Category", ["Large Cap", "Mid Cap", "Small Cap"], horizontal=True)

if category == "Large Cap":
    comp_map = {"TATASTEEL.NS": "Tata Steel", "SAIL.NS": "SAIL"}
elif category == "Mid Cap":
    comp_map = {"HINDALCO.NS": "Hindalco", "NMDC.NS": "NMDC"}
else:
    comp_map = {"MOIL.NS": "MOIL", "JINDALSAW.NS": "Jindal Saw"}

with c2:
    selected_ticker = st.selectbox("Select Company", list(comp_map.keys()), format_func=lambda x: comp_map[x])

with c3:
    days = st.slider("Forecast Days", 1, 30, 7)

st.markdown("</div>", unsafe_allow_html=True)

selected_label = comp_map[selected_ticker]

# Load Data
with st.spinner('Fetching real-time market data...'):
    df, live_price, info = fetch_live_data(selected_ticker)
    
static_vals = STATIC_DATA.get(selected_ticker, {})
analysis_vals = ANALYSIS_DATA.get(selected_ticker, {})

# CALCULATE SMART ROE
smart_roe = calculate_smart_roe(info, selected_ticker)

# Main Content
if not df.empty:
    ml_output = run_analytics(df, days)
    if ml_output is None:
        st.warning("Not enough data for ML prediction comparison.")
        st.stop()

    pred_price = ml_output["future_price"]
    metrics = ml_output["metrics"]
    

    prev_close = df['Close'].iloc[-2] if len(df) > 1 else live_price
    change = live_price - prev_close
    theme_color = "#26a69a" if change >= 0 else "#ef5350"
    pct_change = (change / prev_close) * 100
    arrow = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div style="padding: 20px 0; border-bottom: 1px solid #1a1a1a; margin-bottom: 24px;">
        <div style="font-size: 24px; font-weight: 600; color: #fff; margin-bottom: 8px;">
            📊 {selected_label.upper()}
        </div>
        <div style="display: flex; align-items: baseline; gap: 16px;">
            <span style="font-size: 32px; font-weight: 700; color: #fff;">₹{live_price:,.2f}</span>
            <span style="font-size: 18px; font-weight: 600; color: {theme_color}; padding: 4px 12px; background-color: {theme_color}20; border-radius: 4px;">
                {arrow} {abs(pct_change):.2f}% (₹{abs(change):.2f})
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not info:
        st.warning("⚠️ Live market data unavailable. Displaying cached/static data.")

    # Analysis Toggle
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False
    
    def toggle_analysis():
        st.session_state.show_analysis = not st.session_state.show_analysis

    st.button("🔍 SHOW COMPARATIVE & TREND ANALYSIS", on_click=toggle_analysis)

    # Analysis Section
    if st.session_state.show_analysis:
        with st.container():
            st.markdown('<div class="card" style="border-left: 3px solid #26a69a;">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-header" style="color: #26a69a;">DETAILED ANALYSIS: {selected_label}</div>', unsafe_allow_html=True)
            
            t1, t2, t3 = st.tabs(["📉 TREND ANALYSIS", "⚖️ COMPARATIVE (YoY)", "📊 LIVE RATIOS"])
            
            with t1:
                st.markdown("**10-Year Trend Analysis (Base Year 2016 = 100)**")
                if "trend" in analysis_vals and analysis_vals["trend"] is not None:
                    trend_df = analysis_vals["trend"].set_index("Metric").T.reset_index().rename(columns={"index": "Year"})
                    metrics_to_plot = [c for c in trend_df.columns if c != "Year"]
                    
                    fig_trend = px.line(
                        trend_df, x="Year", y=metrics_to_plot, 
                        markers=True, title="10-Year Trend Index (2016-2025)",
                        color_discrete_sequence=["#26a69a", "#ef5350", "#42a5f5", "#ffa726", "#ab47bc"]
                    )
                    fig_trend.update_layout(
                        height=400, 
                        plot_bgcolor='#0d0d0d', 
                        paper_bgcolor='#0d0d0d', 
                        font=dict(color="#999"),
                        xaxis=dict(showgrid=True, gridcolor='#1a1a1a'),
                        yaxis=dict(showgrid=True, gridcolor='#1a1a1a')
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Trend Data Not Available")
            
            with t2:
                st.markdown("**Comparative Statement (YoY Growth %)**")
                if "trend" in analysis_vals and analysis_vals["trend"] is not None:
                    trend_df = analysis_vals["trend"].set_index("Metric")
                    years = [str(y) for y in range(2017, 2026)]
                    growth_dict = {"Year": years}
                    
                    for metric in trend_df.index:
                        growth_list = []
                        for y in years:
                            prev_y = str(int(y)-1)
                            val = trend_df.loc[metric, y]
                            prev_val = trend_df.loc[metric, prev_y]
                            growth = ((val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
                            growth_list.append(growth)
                        growth_dict[metric] = growth_list
                    
                    growth_df = pd.DataFrame(growth_dict)
                    main_metrics = [m for m in ["Revenue", "EBIT", "Reserves & Surplus"] if m in growth_df.columns]
                    
                    fig_growth = px.bar(
                        growth_df, x="Year", y=main_metrics, barmode='group',
                        title="YoY Growth Percentage (2017-2025)",
                        color_discrete_sequence=["#26a69a", "#42a5f5", "#ffa726"]
                    )
                    fig_growth.update_layout(
                        height=400, 
                        plot_bgcolor='#0d0d0d', 
                        paper_bgcolor='#0d0d0d', 
                        font=dict(color="#999"),
                        xaxis=dict(showgrid=True, gridcolor='#1a1a1a'),
                        yaxis=dict(showgrid=True, gridcolor='#1a1a1a')
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)
                else:
                    st.info("Comparative Data Not Available")

            with t3:
                st.markdown("**Key Financial Ratios (Real-Time)**")
                try:
                    live_ratios = {
                        "P/E Ratio": info.get('trailingPE', 0),
                        "Forward P/E": info.get('forwardPE', 0),
                        "Price/Book": info.get('priceToBook', 0),
                        "Debt/Equity": info.get('debtToEquity', 0),
                        "Current Ratio": info.get('currentRatio', 0)
                    }
                    
                    st.markdown("##### Performance Gauges")
                    c1, c2, c3 = st.columns(3)
                    
                    def make_gauge(title, val, min_v, max_v):
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", 
                            value=val, 
                            title={'text': title, 'font': {'size': 14, 'color': '#999'}}, 
                            gauge={
                                'axis': {'range': [min_v, max_v]}, 
                                'bar': {'color': "#26a69a"}, 
                                'bgcolor': "#1a1a1a", 
                                'borderwidth': 2, 
                                'bordercolor': "#1a1a1a"
                            }
                        ))
                        fig.update_layout(
                            height=200, 
                            margin=dict(l=20,r=20,t=40,b=20), 
                            paper_bgcolor='#0d0d0d', 
                            font=dict(color="#999")
                        )
                        return fig
                    
                    pe_val = live_ratios.get("P/E Ratio", 0) if isinstance(live_ratios.get("P/E Ratio"), (int, float)) else 0
                    
                    # USE SMART ROE HERE
                    roe_val = smart_roe * 100
                    
                    pm_val = (info.get('profitMargins', 0) or 0) * 100

                    with c1:
                        st.plotly_chart(make_gauge("P/E Ratio", pe_val, 0, 50), use_container_width=True)
                    with c2:
                        st.plotly_chart(make_gauge("ROE %", roe_val, -20, 40), use_container_width=True)
                    with c3:
                        st.plotly_chart(make_gauge("Profit Margin %", pm_val, 0, 30), use_container_width=True)
                    
                    st.markdown("##### Detailed Ratios")
                    ratio_df = pd.DataFrame(list(live_ratios.items()), columns=["Ratio", "Value"])
                    st.table(ratio_df)
                    
                except Exception as e:
                    st.error(f"Could not fetch live ratios. Error: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Main Grid
    left_col, right_col = st.columns([1, 2])

    with left_col:
        # Trend Summary
        st.markdown('<div class="card"><div class="card-header">Trend Summary</div>', unsafe_allow_html=True)
        if "trend" in analysis_vals and analysis_vals["trend"] is not None:
            trend_vals = analysis_vals["trend"].loc[0].values[1:] 
            trend_years = [str(y) for y in range(2016, 2026)]
            
            fig_rev = px.area(x=trend_years, y=trend_vals, title="10Y Revenue Trend Index")
            fig_rev.update_traces(line_color='#26a69a', fill='tozeroy')
            fig_rev.update_layout(
                height=150, 
                margin=dict(l=0,r=0,t=30,b=0), 
                paper_bgcolor='#0d0d0d', 
                plot_bgcolor='#0d0d0d', 
                xaxis=dict(showgrid=False, tickfont=dict(color='#666')), 
                yaxis=dict(showgrid=False, visible=False), 
                font=dict(color='#999')
            )
            st.plotly_chart(fig_rev, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(f"<div style='font-size:13px; color:#ccc; margin-top:10px;'><b>Trend:</b> {static_vals.get('trend_txt')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Financial Metrics
        st.markdown('<div class="card"><div class="card-header">Financial Metrics (Live)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{info.get('trailingPE', 'N/A')}</div>
                <div class="metric-lbl">P/E Ratio</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{info.get('trailingEps', 'N/A')}</div>
                <div class="metric-lbl">EPS (TTM)</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{info.get('regularMarketOpen', 'N/A')}</div>
                <div class="metric-lbl">Open Price</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{f"{smart_roe*100:.2f}%"}</div>
                <div class="metric-lbl">ROE %</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        # AI Forecast
        st.markdown('<div class="card"><div class="card-header">AI Predictive Modeling</div>', unsafe_allow_html=True)
        direction = "UP" if ((pred_price - live_price)/live_price) > 0 else "DOWN"
        p_arrow = "▲" if ((pred_price - live_price)/live_price) > 0 else "▼"
        forecast_pct = abs((pred_price - live_price)/live_price)*100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f0f0f, #131313); border: 1px solid #1a1a1a; padding: 20px; margin-bottom: 15px; text-align: center; border-radius: 8px;">
            <div style="color:#666; font-size:11px; text-transform: uppercase; letter-spacing: 0.8px;">Forecast for {days} Days Ahead</div>
            <div style="font-size:36px; color:{theme_color}; font-weight:700; margin: 12px 0;">{p_arrow} {direction} {forecast_pct:.2f}%</div>
            <div style="color:#999; font-size: 14px;">Target Price: <b style="color:#fff;">₹{pred_price:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Performance Heatmap
        hm_z = [[metrics['NN_RMSE'], metrics['NN_MAPE']], [metrics['RF_RMSE'], metrics['RF_MAPE']]]
        fig_hm = go.Figure(go.Heatmap(
            z=hm_z, 
            x=['RMSE (₹)', 'MAPE (%)'], 
            y=['Neural Network', 'Random Forest'], 
            colorscale='RdYlGn_r', 
            texttemplate="%{z:.2f}", 
            textfont={"size": 14, "color": "#fff"},
            showscale=False
        ))
        fig_hm.update_layout(
            height=180, 
            margin=dict(l=0,r=0,t=0,b=0), 
            paper_bgcolor='#0d0d0d', 
            plot_bgcolor='#0d0d0d',
            xaxis=dict(side='top', tickfont=dict(color='#999')),
            yaxis=dict(tickfont=dict(color='#999'))
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
            <div class="card">
            <div class="card-header">Random Forest – Actual vs Predicted</div>
            """, unsafe_allow_html=True)

        fig_rf = go.Figure()

        fig_rf.add_trace(go.Scatter(
            x=ml_output["dates"],
            y=ml_output["actual"],
            mode="lines",
            name="Actual Price",
            line=dict(color="#cccccc", width=2)
        ))

        fig_rf.add_trace(go.Scatter(
            x=ml_output["dates"],
            y=ml_output["rf_pred"],
            mode="lines",
            name="RF Predicted",
            line=dict(color="#00ff00", width=2, dash="dot")
        ))

        fig_rf.update_layout(
            height=280,
            paper_bgcolor="#111",
            plot_bgcolor="#111",
            font=dict(color="#ddd"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=-0.25)
        )

        st.plotly_chart(fig_rf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <div class="card-header">Neural Network – Actual vs Predicted</div>
        """, unsafe_allow_html=True)

        fig_nn = go.Figure()

        fig_nn.add_trace(go.Scatter(
        x=ml_output["dates"],
        y=ml_output["actual"],
        mode="lines",
        name="Actual Price",
        line=dict(color="#cccccc", width=2)
        ))

        fig_nn.add_trace(go.Scatter(
        x=ml_output["dates"],
        y=ml_output["nn_pred"],
        mode="lines",
        name="NN Predicted",
        line=dict(color="#00ceff", width=2, dash="dot")
        ))

        fig_nn.update_layout(
        height=280,
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font=dict(color="#ddd"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=-0.25)
        )   

        st.plotly_chart(fig_nn, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


        # AI Recommendation
        st.markdown('<div class="card"><div class="card-header">AI Recommendation</div>', unsafe_allow_html=True)
        diff_pct = ((pred_price - live_price) / live_price) * 100
        
        if diff_pct > 2.0:
            signal = "BUY"
            sig_color = "#26a69a"
            desc = f"Strong upside potential of {diff_pct:.2f}% projected."
        elif diff_pct < -2.0:
            signal = "SELL"
            sig_color = "#ef5350"
            desc = f"Downside risk of {abs(diff_pct):.2f}% projected."
        else:
            signal = "HOLD"
            sig_color = "#ffa726"
            desc = f"Price stable. Projected change ({diff_pct:.2f}%) is within noise."
        
        st.markdown(f"""
        <div style="text-align: center; padding: 24px 0;">
            <div style="display: inline-block; padding: 12px 32px; font-size: 24px; font-weight: 700; letter-spacing: 1px; 
                        color: {sig_color}; background-color: {sig_color}20; border: 2px solid {sig_color}; border-radius: 8px;">
                {signal}
            </div>
            <div style="color: #999; font-size: 13px; margin-top: 16px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Live Market Chart (Full Width)
    st.markdown('<div class="card"><div class="card-header">Live Market Price Action</div>', unsafe_allow_html=True)

    if 'Open' in df.columns:
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        )])
    else:
        fig = px.line(df, x='Date', y='Close')

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#0d0d0d',
        font=dict(color="#999"),
        xaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            rangeslider_visible=False,
            zeroline=False,
            color='#666'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1a1a1a',
            zeroline=False,
            color='#666'
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Ticker Tape
    @st.cache_data(ttl=300)
    def fetch_ticker_content(symbol, label):
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")

        price_part = f"{label} • PRICE N/A"
        if not hist.empty:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else curr
            chg = ((curr - prev) / prev) * 100
            arrow = "▲" if chg >= 0 else "▼"
            col = "#26a69a" if chg >= 0 else "#ef5350"

            price_part = (
                f"<span style='color:#666'>{label}</span> "
                f"<span style='color:#fff'>₹{curr:,.2f}</span> "
                f"<span style='color:{col}'>{arrow} {abs(chg):.2f}%</span>"
            )

        company_name = COMPANY_NAMES.get(symbol, label)
        news_list = fetch_company_news(symbol, company_name)

        if news_list and news_list[0] != "News unavailable":
            news_part = " | " + " • ".join(news_list[:3])
        else:
            news_part = " | No fresh headlines"

        return price_part + news_part

    tape_content = fetch_ticker_content(selected_ticker, selected_label)
    full_tape = tape_content + "   " + tape_content

    st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-content">{full_tape}</div>
    </div>
    """, unsafe_allow_html=True)
