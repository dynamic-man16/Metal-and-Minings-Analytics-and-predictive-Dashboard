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
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Metal and minings Sector Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --- 2. CSS STYLING (Red/Black Theme) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #050505; }
    
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto Condensed', sans-serif; }

    /* Grid Layout */
    .dashboard-container {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 20px;
        padding: 10px;
    }
    
    /* Card Styling */
    .card {
        background-color: #111; 
        border: 1px solid #333;
        border-left: 3px solid #cc0000; 
        border-radius: 4px; 
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    .card-header {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        color: #cc0000; 
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
        letter-spacing: 1px;
    }
    
    /* Metric Boxes */
    .metric-row { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .metric-box { text-align: center; }
    .metric-val { font-size: 22px; font-weight: 700; color: #eee; }
    .metric-lbl { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Ticker */
    .ticker-wrap {
        position: fixed; bottom: 0; left: 0; width: 100%;
        height: 45px; background-color: #000; 
        border-top: 2px solid #cc0000; 
        z-index: 999999;
        overflow: hidden; white-space: nowrap;
        display: flex; align-items: center;
    }
    .ticker-content {
        display: inline-block;
        font-family: 'Roboto Condensed', monospace;
        font-size: 16px; color: #f0f0f0; font-weight: 400;
        white-space: nowrap;
        padding-left: 100%;
        animation: ticker-scroll 45s linear infinite;
    }
    @keyframes ticker-scroll { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-100%, 0, 0); } }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #1a1a1a;
        color: #cc0000;
        border: 1px solid #cc0000;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #cc0000;
        color: white;
        border-color: #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HARDCODED STATIC DATA (Existing) ---
STATIC_DATA = {
    "TATASTEEL.NS": {
        "trend_txt": "Revenue Consolidation: -5.9% dip in 2025 vs 2024.",
        "comp_txt": "EBITDA Margin: Outperforming Sector Average (14% vs 11%)."
    },
    "SAIL.NS": {
        "trend_txt": "Stable Revenue > ₹1L Cr. Slight dip in 2025 (-2.7%).",
        "comp_txt": "Liquidity: Strongest Cash Flow in Public Sector Steel."
    },
    "HINDALCO.NS": {
        "trend_txt": "Strong Growth: +12% Revenue Jump in 2025.",
        "comp_txt": "Growth Leader: Highest Top-line growth among peers."
    },
    "NMDC.NS": {
        "trend_txt": "Robust Expansion: +11% Revenue Growth YoY.",
        "comp_txt": "Valuation: Lowest P/E (9.5X) suggests deep value."
    },
    "MOIL.NS": {
        "trend_txt": "Steady Incline: Consistent ~9% YoY Growth.",
        "comp_txt": "Dividends: Highest yield potential in small-cap mining."
    },
    "JINDALSAW.NS": {
        "trend_txt": "Plateauing Revenue: Flat growth in 2025 vs 2024.",
        "comp_txt": "Profitability: Highest EPS (₹29.44) in peer group."
    }
}

# --- 4. EXTENDED ANALYSIS DATA (P&L + BALANCE SHEET PARAMETERS) ---
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

@st.cache_data(ttl=300)
def fetch_live_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y", interval="1d")
        if hist.empty: hist = stock.history(period="max", interval="1d")
        try:
            todays_data = stock.history(period="1d", interval="1m")
            current_price = todays_data['Close'].iloc[-1] if not todays_data.empty else hist['Close'].iloc[-1]
        except:
            current_price = hist['Close'].iloc[-1]
        hist.reset_index(inplace=True)
        return hist, current_price, stock.info
    except:
        return pd.DataFrame(), 0.0, {}

# --- 5. ML ENGINE ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_analytics(df, days_forecast):
    if len(df) < 100: return 0.0, {"NN_RMSE":0,"NN_MAPE":0,"RF_RMSE":0,"RF_MAPE":0}
    df = df.copy()
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df.dropna(inplace=True)
    X = df[['Date_Ordinal', 'MA_50', 'EMA_20', 'RSI', 'Lag_1', 'Lag_2']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)).fit(X_train, y_train)
    rf_pred, nn_pred = rf.predict(X_test), nn.predict(X_test)
    metrics = {
        "NN_RMSE": np.sqrt(mean_squared_error(y_test, nn_pred)), "NN_MAPE": np.mean(np.abs((y_test - nn_pred) / y_test)) * 100,
        "RF_RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)), "RF_MAPE": np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
    }
    last_row = df.iloc[-1]
    next_ord = df['Date'].iloc[-1].toordinal() + days_forecast
    inp = pd.DataFrame([{'Date_Ordinal': next_ord, 'MA_50': last_row['MA_50'], 'EMA_20': last_row['EMA_20'], 'RSI': last_row['RSI'], 'Lag_1': last_row['Close'], 'Lag_2': last_row['Lag_1']}])
    future_price = (rf.predict(inp)[0] + nn.predict(inp)[0]) / 2
    return future_price, metrics

# --- 6. APP LAYOUT ---
category = st.sidebar.radio("Category", ["Large Cap", "Mid Cap", "Small Cap"])
if category == "Large Cap": comp_map = {"TATASTEEL.NS": "Tata Steel", "SAIL.NS": "SAIL"}
elif category == "Mid Cap": comp_map = {"HINDALCO.NS": "Hindalco", "NMDC.NS": "NMDC"}
else: comp_map = {"MOIL.NS": "MOIL", "JINDALSAW.NS": "Jindal Saw"}
selected_ticker = st.sidebar.selectbox("Select Company", list(comp_map.keys()), format_func=lambda x: comp_map[x])
selected_label = comp_map[selected_ticker]
days = st.sidebar.slider("Forecast Days", 1, 30, 7)

df, live_price, info = fetch_live_data(selected_ticker)
static_vals = STATIC_DATA.get(selected_ticker, {})
analysis_vals = ANALYSIS_DATA.get(selected_ticker, {})

if not df.empty:
    pred_price, metrics = run_analytics(df, days)
    prev_close = df['Close'].iloc[-2]
    change = live_price - prev_close
    theme_color = "#00ff00" if change >= 0 else "#ff3333"

    st.markdown(f"## 📊 Metal and Minings SECTOR ANALYTICS: <span style='color:{theme_color}'>{selected_label.upper()}</span>", unsafe_allow_html=True)

    # --- ANALYSIS TOGGLE BUTTON ---
    if "show_analysis" not in st.session_state: st.session_state.show_analysis = False
    
    def toggle_analysis():
        st.session_state.show_analysis = not st.session_state.show_analysis

    st.button("🔍 SHOW COMPARATIVE & TREND ANALYSIS", on_click=toggle_analysis)

    # --- ANALYSIS CONTAINER ---
    if st.session_state.show_analysis:
        with st.container():
            st.markdown(f"""<div class="card" style="border-left: 3px solid #00ceff;">
                            <div class="card-header" style="color: #00ceff;">DETAILED ANALYSIS: {selected_label}</div>""", unsafe_allow_html=True)
            
            t1, t2, t3 = st.tabs(["📉 TREND ANALYSIS (GRAPHICAL)", "⚖️ COMPARATIVE (YoY GRAPH)", "📊 RATIOS (LIVE)"])
            
            with t1:
                st.markdown("**10-Year Trend Analysis (Base Year 2016 = 100)**")
                if "trend" in analysis_vals and analysis_vals["trend"] is not None:
                    # Convert to Long Format for Plotly
                    trend_df = analysis_vals["trend"].set_index("Metric").T.reset_index().rename(columns={"index": "Year"})
                    # Line Chart for Trend (INCLUDES NEW BS PARAMS)
                    # We dynamically select all columns that are not 'Year'
                    metrics_to_plot = [c for c in trend_df.columns if c != "Year"]
                    
                    fig_trend = px.line(trend_df, x="Year", y=metrics_to_plot, 
                                        markers=True, title="10-Year Trend Index (2016-2025)",
                                        color_discrete_sequence=["#00ff00", "#ff3333", "#00ceff", "#ffff00", "#ff00ff"])
                    fig_trend.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#ddd"))
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Trend Data Not Available")
            
            with t2:
                st.markdown("**Comparative Statement (YoY Growth %)**")
                if "trend" in analysis_vals and analysis_vals["trend"] is not None:
                    # Calculate YoY Growth dynamically from Trend Data
                    trend_df = analysis_vals["trend"].set_index("Metric")
                    years = [str(y) for y in range(2017, 2026)]
                    growth_dict = {"Year": years}
                    
                    # Calculate for all available metrics
                    for metric in trend_df.index:
                        growth_list = []
                        for y in years:
                            prev_y = str(int(y)-1)
                            val = trend_df.loc[metric, y]
                            prev_val = trend_df.loc[metric, prev_y]
                            growth = ((val - prev_val) / prev_val) * 100
                            growth_list.append(growth)
                        growth_dict[metric] = growth_list
                    
                    growth_df = pd.DataFrame(growth_dict)
                    
                    # Bar Chart for Growth (Top 3 Metrics for clarity)
                    main_metrics = [m for m in ["Revenue", "EBIT", "Reserves & Surplus"] if m in growth_df.columns]
                    
                    fig_growth = px.bar(growth_df, x="Year", y=main_metrics, barmode='group',
                                        title="YoY Growth Percentage (2017-2025)",
                                        color_discrete_sequence=["#00ff00", "#00ceff", "#ffff00"])
                    fig_growth.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#ddd"))
                    st.plotly_chart(fig_growth, use_container_width=True)
                else:
                    st.info("Comparative Data Not Available")

            with t3:
                st.markdown("**Key Financial Ratios (Real-Time Live Data)**")
                try:
                    live_ratios = {
                        "P/E Ratio": info.get('trailingPE', 0),
                        "Forward P/E": info.get('forwardPE', 0),
                        "Price/Book": info.get('priceToBook', 0),
                        "Debt/Equity": info.get('debtToEquity', 0),
                        "Current Ratio": info.get('currentRatio', 0)
                    }
                    
                    # Gauge Charts for Ratios in their own dedicated container/columns
                    st.markdown("##### Performance Gauges")
                    c1, c2, c3 = st.columns(3)
                    
                    def make_gauge(title, val, min_v, max_v):
                        # Create gauge with specific layout to ensure alignment
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number", 
                            value = val, 
                            title = {'text': title, 'font': {'size': 14}}, 
                            gauge = {
                                'axis': {'range': [min_v, max_v]}, 
                                'bar': {'color': "#cc0000"}, 
                                'bgcolor': "#333",
                                'borderwidth': 2,
                                'bordercolor': "#333"
                            }
                        ))
                        # Fix height and margins
                        fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#ddd"))
                        return fig
                    
                    # Ensure values are numeric before plotting
                    pe_val = live_ratios["P/E Ratio"] if isinstance(live_ratios["P/E Ratio"], (int, float)) else 0
                    roe_val = (info.get('returnOnEquity', 0) or 0) * 100
                    pm_val = (info.get('profitMargins', 0) or 0) * 100

                    with c1: st.plotly_chart(make_gauge("P/E Ratio", pe_val, 0, 50), use_container_width=True)
                    with c2: st.plotly_chart(make_gauge("ROE %", roe_val, -20, 40), use_container_width=True)
                    with c3: st.plotly_chart(make_gauge("Profit Margin %", pm_val, 0, 30), use_container_width=True)
                    
                    # Table for details in a separate section below gauges
                    st.markdown("##### Detailed Ratios Table")
                    ratio_df = pd.DataFrame(list(live_ratios.items()), columns=["Ratio", "Current Value"])
                    st.table(ratio_df)
                except Exception as e:
                    st.error(f"Could not fetch live ratios: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # --- MAIN GRID ---
    left_col, right_col = st.columns([1, 2])

    with left_col:
        # --- BOX 1: COMPARATIVE & TREND (SUMMARY) ---
        st.markdown("""<div class="card"><div class="card-header">Trend Summary</div>""", unsafe_allow_html=True)
        trend_vals = analysis_vals["trend"].loc[0].values[1:] 
        trend_years = [str(y) for y in range(2016, 2026)]
        fig_rev = px.area(x=trend_years, y=trend_vals, title="10Y Revenue Trend Index")
        fig_rev.update_traces(line_color='#cc0000', fill='tozeroy')
        fig_rev.update_layout(height=150, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, tickfont=dict(color='#888')), yaxis=dict(showgrid=False, visible=False), font=dict(color='#ccc'))
        st.plotly_chart(fig_rev, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<div style='font-size:13px; color:#ddd; margin-top:10px;'><b>Trend:</b> {static_vals.get('trend_txt')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 2: METRICS (UNCHANGED) ---
        st.markdown("""<div class="card"><div class="card-header">Financial Metrics (Live)</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{info.get('trailingPE', 'N/A')}</div><div class="metric-lbl">P/E RATIO</div></div>
            <div class="metric-box"><div class="metric-val">{info.get('trailingEps', 'N/A')}</div><div class="metric-lbl">EPS (TTM)</div></div>
        </div>
        <div class="metric-row">
             <div class="metric-box"><div class="metric-val">{info.get('regularMarketOpen', 'N/A')}</div><div class="metric-lbl">OPEN PRICE</div></div>
             <div class="metric-box"><div class="metric-val">{f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A'}</div><div class="metric-lbl">ROE %</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        # --- BOX 3: AI PREDICTION ---
        st.markdown("""<div class="card"><div class="card-header">AI Predictive Modeling</div>""", unsafe_allow_html=True)
        direction = "UP" if ((pred_price - live_price)/live_price) > 0 else "DOWN"
        p_arrow = "▲" if ((pred_price - live_price)/live_price) > 0 else "▼"
        st.markdown(f"""<div style="background: #000; border: 1px solid #333; padding: 15px; margin-bottom: 15px; text-align: center;">
            <div style="color:#888; font-size:12px;">FORECAST FOR {days} DAYS AHEAD</div>
            <div style="font-size:32px; color:{theme_color}; font-weight:bold;">{p_arrow} {direction} {abs((pred_price - live_price)/live_price)*100:.2f}%</div>
            <div style="color:#ccc;">Target Price: <b>₹{pred_price:,.2f}</b></div>
        </div>""", unsafe_allow_html=True)
        hm_z = [[metrics['NN_RMSE'], metrics['NN_MAPE']], [metrics['RF_RMSE'], metrics['RF_MAPE']]]
        fig_hm = go.Figure(go.Heatmap(z=hm_z, x=['RMSE (₹)', 'MAPE (%)'], y=['Neural Network', 'Random Forest'], colorscale='RdYlGn_r', texttemplate="%{z:.2f}", textfont={"size": 14}))
        fig_hm.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 3.5: AI RECOMMENDATION (NEW) ---
        st.markdown("""<div class="card"><div class="card-header">AI Recommendation</div>""", unsafe_allow_html=True)
        
        # Calculate percentage difference
        diff_pct = ((pred_price - live_price) / live_price) * 100
        
        # Determine Signal
        if diff_pct > 2.0:
            signal = "BUY"
            sig_color = "#00ff00" # Green
            desc = f"Strong upside potential of {diff_pct:.2f}% projected."
        elif diff_pct < -2.0:
            signal = "SELL"
            sig_color = "#ff3333" # Red
            desc = f"Downside risk of {abs(diff_pct):.2f}% projected."
        else:
            signal = "HOLD"
            sig_color = "#ffcc00" # Yellow
            desc = f"Price stable. Projected change ({diff_pct:.2f}%) is within noise."
            
        st.markdown(f"""
        <div style="text-align: center; padding: 10px;">
            <div style="font-size: 36px; font-weight: 800; color: {sig_color}; letter-spacing: 2px;">{signal}</div>
            <div style="color: #ccc; font-size: 14px; margin-top: 5px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BOX 4: CHART ---
        st.markdown("""<div class="card"><div class="card-header">Live Market Chart</div>""", unsafe_allow_html=True)
        chart_df = df.tail(100)
        fig = go.Figure(data=[go.Candlestick(x=chart_df['Date'], open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], increasing_line_color='#00ff00', decreasing_line_color='#ff3333')])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='#1c1f26', plot_bgcolor='#1c1f26', font=dict(color="#eee"), xaxis_rangeslider_visible=False, xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- 7. TICKER ---
@st.cache_data(ttl=300) 
def fetch_selected_ticker_content(symbol, name):
    try:
        stock = yf.Ticker(symbol)
        
        # 1. Price Data
        hist = stock.history(period="5d")
        if not hist.empty:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[-1]
            chg = ((curr - prev) / prev) * 100
            arrow = "▲" if chg >= 0 else "▼"
            col = "#00ff00" if chg >= 0 else "#ff3333"
            price_str = f"🔴 {name} LIVE: <span style='color:{col}'>₹{curr:,.2f} ({arrow} {chg:.2f}%)</span>"
        else:
            price_str = f"🔴 {name}: PRICE N/A"
            
        # 2. News Data - Robust Fetching
        news_str = ""
        try:
            news_list = stock.news
            if news_list:
                headlines = []
                for n in news_list[:5]: # Get top 5
                    # Try multiple keys as yfinance schema can change
                    title = n.get('title') or n.get('headline')
                    if title: headlines.append(title)
                
                if headlines:
                    news_str = "   &nbsp;&nbsp;&nbsp;  📰  NEWS: " + "  •  ".join(headlines)
        except Exception:
            pass 
        
        if not news_str:
            # Fallback if API returns empty but valid response
            news_str = "   &nbsp;&nbsp;&nbsp;  📰  NEWS: Market data available. Check local news sources for latest updates."

        return f"{price_str} {news_str}"
        
    except Exception as e:
        return f"🔴 {name}: Data Unavailable ({str(e)})"

tape_content = fetch_selected_ticker_content(selected_ticker, selected_label)
full_tape = (tape_content + "   &nbsp;&nbsp;&nbsp;   ") * 5 
st.markdown(f"""<div class="ticker-wrap"><div class="ticker-content">{full_tape}</div></div>""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
