import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import os
from io import BytesIO

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Cerebrum Forecasting",
    page_icon="📈",
    layout="wide"
)

# Custom CSS - Enhanced
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .event-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 5px solid #ccc;
        background-color: #fff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1f2937; /* Force dark text color for visibility */
    }
    .event-card h4 {
        color: #111827; /* Force darker header color */
        margin-top: 0;
        font-weight: bold;
    }
    .event-card p {
        color: #374151; /* Force dark paragraph color */
        margin-bottom: 0.5rem;
    }
    .pra-event { border-left-color: #f59e0b; background-color: #fffbeb; }
    .peak-event { border-left-color: #ef4444; background-color: #fef2f2; }
    .pasca-event { border-left-color: #10b981; background-color: #ecfdf5; }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 Cerebrum Multivariate Forecasting System</p>', unsafe_allow_html=True)
st.markdown("**Sistem Prediksi Cerdas: Deteksi Pola Pra-Event, Peak Time, & Pasca-Event**")

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def generate_dummy_data():
    """Generate comprehensive dummy data with multiple categories"""
    date_range = pd.date_range(start='2023-01-01', end='2025-12-31', freq='H')
    np.random.seed(42)
    n = len(date_range)
    
    # Generate base patterns
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / (24*365))
    
    # 1. User Creation Pattern (Base)
    user_trend = np.linspace(5, 15, n)
    user_noise = np.random.normal(0, 3, n)
    user_counts = np.maximum(user_trend + seasonality/2 + user_noise, 0)
    
    # 2. Activity/Attempt Pattern (Correlated with Users but higher volume)
    activity_trend = user_trend * 3
    activity_noise = np.random.normal(0, 10, n)
    activity_counts = np.maximum(activity_trend + seasonality + activity_noise, 0)
    
    # 3. Transaction Pattern (Lagged correlation with Activity)
    trans_trend = activity_trend * 0.2
    trans_noise = np.random.normal(0, 2, n)
    trans_counts = np.maximum(trans_trend + seasonality*0.1 + trans_noise, 0)
    
    # Add Events (Spikes) affecting all
    events_dates = pd.to_datetime(['2023-06-15', '2023-12-01', '2024-06-20', '2024-12-10', '2025-06-25'])
    for evt in events_dates:
        indices = np.where((date_range >= evt - timedelta(days=2)) & (date_range <= evt + timedelta(days=2)))[0]
        user_counts[indices] *= 2.0
        activity_counts[indices] *= 3.5
        trans_counts[indices] *= 2.5
        
    records = []
    sample_idx = np.random.choice(n, size=int(n*0.3), replace=False)
    sample_idx.sort()
    
    for i in sample_idx:
        dt = date_range[i]
        
        for _ in range(int(user_counts[i] / 5)):
            records.append([dt, 'User'])
            
        for _ in range(int(activity_counts[i] / 5)):
            records.append([dt, 'Aktivitas'])
            
        for _ in range(int(trans_counts[i] / 5)):
            records.append([dt, 'Transaksi'])
            
    return pd.DataFrame(records, columns=['created_at', 'kategori'])

@st.cache_data(show_spinner=False)
def load_local_csv(file_path):
    """Load large CSV from local path with caching and error handling"""
    try:
        # Try reading with different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode file with any standard encoding")
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def process_multivariate_data(df):
    """
    Pivot data to create columns for each category per day.
    Enhanced with data validation and cleaning.
    """
    # Validate required columns
    if 'created_at' not in df.columns or 'kategori' not in df.columns:
        st.error("❌ Dataset harus memiliki kolom 'created_at' dan 'kategori'")
        return None
    
    # Ensure datetime with better error handling
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Remove invalid dates
    initial_len = len(df)
    df = df.dropna(subset=['created_at'])
    if len(df) < initial_len:
        st.warning(f"⚠️ {initial_len - len(df)} baris dengan tanggal invalid dihapus")
    
    df['date'] = df['created_at'].dt.date
    
    # Pivot: Index=Date, Columns=Kategori, Values=Count
    daily_pivot = df.groupby(['date', 'kategori']).size().unstack(fill_value=0)
    daily_pivot.index = pd.to_datetime(daily_pivot.index)
    
    # Sort index
    daily_pivot = daily_pivot.sort_index()
    
    # Add Total column
    daily_pivot['Total'] = daily_pivot.sum(axis=1)
    
    # Fill missing dates with 0
    full_range = pd.date_range(start=daily_pivot.index.min(), end=daily_pivot.index.max(), freq='D')
    daily_pivot = daily_pivot.reindex(full_range, fill_value=0)
    
    # Ensure index name is clear for later use
    daily_pivot.index.name = 'date'
    
    return daily_pivot

def detect_event_phases(forecast_df, threshold_multiplier=1.5, prep_days=7, post_days=7):
    """
    Identifies Pra-Event, Peak, and Pasca-Event phases based on forecast.
    Enhanced with better edge case handling.
    """
    if forecast_df is None or len(forecast_df) == 0:
        return [], 0
        
    df = forecast_df.copy()
    mean_val = df['yhat'].mean()
    std_val = df['yhat'].std()
    
    # Handle edge case where std is 0
    if std_val == 0:
        st.warning("⚠️ Data terlalu uniform untuk deteksi event otomatis")
        return [], mean_val
    
    threshold = mean_val + (threshold_multiplier * std_val)
    
    # Identify Peak Days
    df['is_peak'] = df['yhat'] > threshold
    
    if df['is_peak'].sum() == 0:
        return [], threshold
    
    # Group consecutive peak days
    df['peak_group'] = (df['is_peak'] != df['is_peak'].shift()).cumsum()
    peaks = df[df['is_peak']].groupby('peak_group').agg(
        start_date=('ds', 'min'),
        end_date=('ds', 'max'),
        peak_val=('yhat', 'max')
    ).reset_index(drop=True)
    
    event_phases = []
    
    for _, row in peaks.iterrows():
        # Pra Event
        pra_start = row['start_date'] - timedelta(days=prep_days)
        pra_end = row['start_date'] - timedelta(days=1)
        
        # Pasca Event
        pasca_start = row['end_date'] + timedelta(days=1)
        pasca_end = row['end_date'] + timedelta(days=post_days)
        
        event_phases.append({
            'pra_start': pra_start,
            'pra_end': pra_end,
            'peak_start': row['start_date'],
            'peak_end': row['end_date'],
            'pasca_start': pasca_start,
            'pasca_end': row['end_date'] + timedelta(days=post_days),
            'peak_value': row['peak_val']
        })
        
    return event_phases, threshold

def calculate_metrics(actual, predicted):
    """Calculate comprehensive accuracy metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    # WMAPE
    sum_actuals = actual.sum()
    sum_errors = np.abs(actual - predicted).sum()
    wmape = (sum_errors / sum_actuals) * 100 if sum_actuals > 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'WMAPE': wmape,
        'Accuracy': max(0, 100 - wmape)
    }

def export_forecast_results(forecast_df, phases, target_col):
    """Export forecast and phases to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Forecast Data
        forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
        
        # Sheet 2: Event Phases
        if phases:
            phases_df = pd.DataFrame(phases)
            phases_df.to_excel(writer, sheet_name='Event Phases', index=False)
        
        # Sheet 3: Summary Stats
        summary = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                forecast_df['yhat'].mean(),
                forecast_df['yhat'].median(),
                forecast_df['yhat'].std(),
                forecast_df['yhat'].min(),
                forecast_df['yhat'].max()
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

# --- Session State Initialization ---
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_daily' not in st.session_state:
    st.session_state['df_daily'] = None
if 'data_source_mode' not in st.session_state:
    st.session_state['data_source_mode'] = None
if 'custom_holidays' not in st.session_state:
    st.session_state['custom_holidays'] = []
if 'forecast_model' not in st.session_state:
    st.session_state['forecast_model'] = None
if 'forecast_result' not in st.session_state:
    st.session_state['forecast_result'] = None
if 'future_target_df' not in st.session_state:
    st.session_state['future_target_df'] = None
if 'model_params' not in st.session_state:
    st.session_state['model_params'] = {}

# --- Sidebar ---

with st.sidebar:
    st.header("⚙️ Data Configuration")
    
    data_source = st.radio(
        "Sumber Data:", 
        ["📂 Data Demo", "☁️ Upload CSV", "🖥️ File Lokal (>200MB)"]
    )
    
    # Reset data if source changes
    if st.session_state['data_source_mode'] != data_source:
        st.session_state['data_source_mode'] = data_source
        st.session_state['df_raw'] = None
    
    if data_source == "☁️ Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV (dengan kolom 'created_at' dan 'kategori')", type=['csv'])
        if uploaded_file:
            try:
                st.session_state['df_raw'] = pd.read_csv(uploaded_file)
                st.success(f"✅ File '{uploaded_file.name}' berhasil dimuat")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            
    elif data_source == "🖥️ File Lokal (>200MB)":
        st.info("ℹ️ Mode ini membaca file langsung dari folder server untuk bypass limit browser.")
        default_path = "folder/dataset.csv"
        local_path = st.text_input("Path File CSV:", default_path)
        
        if st.button("📥 Muat Data Lokal"):
            if os.path.exists(local_path):
                with st.spinner("Memuat data besar..."):
                    loaded_df = load_local_csv(local_path)
                    if loaded_df is not None:
                        st.session_state['df_raw'] = loaded_df
                        st.success("✅ Data berhasil dimuat!")
            else:
                st.error(f"❌ File tidak ditemukan: {local_path}")
        
        if st.session_state['df_raw'] is not None:
            st.success("✅ Data Lokal Tersimpan di Memori")

    else:  # Demo
        if st.session_state['df_raw'] is None:
            st.session_state['df_raw'] = generate_dummy_data()

    # --- PROCESS DATA IF AVAILABLE ---
    if st.session_state['df_raw'] is not None:
        with st.spinner("Memproses struktur data..."):
            df_daily = process_multivariate_data(st.session_state['df_raw'])
            if df_daily is not None:
                st.session_state['df_daily'] = df_daily
        
        if st.session_state['df_daily'] is not None:
            st.success(f"✅ Data: {len(st.session_state['df_daily'])} hari | {len(st.session_state['df_raw'])} records")
            st.markdown("---")
            
            st.header("🎯 Forecasting Strategy")
            
            # --- ADVANCED SETTINGS ---
            with st.expander("⚙️ Advanced Model Settings", expanded=False):
                changepoint_scale = st.slider(
                    "Changepoint Prior Scale",
                    0.001, 0.5, 0.05, 0.001,
                    help="Fleksibilitas model terhadap perubahan trend (lebih tinggi = lebih fleksibel)"
                )
                seasonality_scale = st.slider(
                    "Seasonality Prior Scale",
                    0.01, 10.0, 10.0, 0.1,
                    help="Kekuatan pola musiman (lebih tinggi = pola musiman lebih kuat)"
                )
                st.session_state['model_params'] = {
                    'changepoint_prior_scale': changepoint_scale,
                    'seasonality_prior_scale': seasonality_scale
                }
            
            # --- PANDUAN PEMILIHAN VARIABEL ---
            with st.expander("📚 Panduan: Apa yang harus saya pilih?", expanded=False):
                st.info("""
                **1. Target Utama (Output):**
                Pilih metrik yang ingin Anda prediksi masa depannya.
                - Pilih **'Transaksi'** untuk revenue/omset
                - Pilih **'Aktivitas'** untuk beban server/traffic
                - Pilih **'User'** untuk pertumbuhan user baru
                
                **2. Prediktor (Input):**
                Pilih variabel yang **mempengaruhi** target tersebut.
                - Jika Target = **'Transaksi'**, Prediktor: **'User'** dan **'Aktivitas'**
                - Jika Target = **'Total'**, kosongkan prediktor (Univariate)
                
                **Tips:** Lihat tab "Analisis Korelasi" untuk menemukan hubungan terkuat!
                """)
            
            available_cols = list(st.session_state['df_daily'].columns)
            
            # 1. Select Target
            target_col = st.selectbox(
                "🎯 Target Utama (yang ingin diprediksi):",
                available_cols,
                index=available_cols.index('Total') if 'Total' in available_cols else 0
            )
            
            # 2. Select Regressors
            potential_regressors = [c for c in available_cols if c != target_col and c != 'Total']
            
            # Smart default selection based on target
            default_regressors = []
            if target_col == 'Transaksi':
                default_regressors = [c for c in potential_regressors if c in ['User', 'Aktivitas']]
            
            selected_regressors = st.multiselect(
                "📊 Prediktor (variabel pendukung):",
                potential_regressors,
                default=default_regressors
            )
            
            st.markdown("---")
            
            # 3. Year Selection
            min_year = st.session_state['df_daily'].index.year.min()
            max_year = st.session_state['df_daily'].index.year.max()
            
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                target_year = st.selectbox(
                    "📅 Tahun Target:", 
                    range(min_year, max_year + 3),
                    index=min(len(range(min_year, max_year + 3))-1, 1)
                )
            
            with col_y2:
                is_backtesting = target_year <= max_year
                if is_backtesting:
                    st.info(f"🔍 Mode: **Backtesting** (data {target_year} tersedia)")
                else:
                    st.info(f"🔮 Mode: **Forecasting** (prediksi masa depan)")

            # 4. Event Settings
            with st.expander("⚙️ Konfigurasi Deteksi Event", expanded=True):
                threshold_mult = st.slider(
                    "Sensitivitas Peak (StdDev)", 
                    0.5, 3.0, 1.5, 0.1,
                    help="Lebih rendah = lebih sensitif (deteksi lebih banyak event)"
                )
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    prep_days = st.slider(
                        "Durasi Pra-Event (Hari)", 
                        1, 30, 14,
                        help="Berapa hari sebelum event untuk persiapan?"
                    )
                with col_p2:
                    post_days = st.slider(
                        "Durasi Pasca-Event (Hari)", 
                        1, 30, 7,
                        help="Berapa hari monitoring setelah event?"
                    )

            # 5. MANUAL HOLIDAY INPUT
            with st.expander("📅 Input Jadwal Event Resmi (Opsional)", expanded=False):
                st.info("""
                **PENTING:** Input jadwal resmi di sini agar model mengenali pola persiapan sebelum event.
                Sistem akan otomatis menandai masa persiapan berdasarkan durasi yang Anda tentukan.
                """)
                
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    h_name = st.text_input("Nama Event", "Event CPNS", key="holiday_name")
                with col_h2:
                    h_dates = st.date_input(
                        "Rentang Tanggal Event", 
                        [], 
                        key="holiday_dates",
                        help="Pilih tanggal mulai dan selesai event"
                    )
                
                if st.button("➕ Tambah Jadwal"):
                    if len(h_dates) == 2:
                        st.session_state['custom_holidays'].append({
                            'holiday': h_name,
                            'ds_start': h_dates[0],
                            'ds_end': h_dates[1]
                        })
                        st.success(f"✅ Jadwal '{h_name}' ditambahkan!")
                        st.rerun()
                    else:
                        st.error("❌ Pilih 2 tanggal (start & end)")
                
                # Show added holidays
                if st.session_state['custom_holidays']:
                    st.markdown("##### 📋 Jadwal Tersimpan:")
                    holidays_display = pd.DataFrame(st.session_state['custom_holidays'])
                    st.dataframe(holidays_display, use_container_width=True, hide_index=True)
                    
                    if st.button("🗑️ Reset Semua Jadwal"):
                        st.session_state['custom_holidays'] = []
                        st.rerun()

# --- Main Logic ---

if st.session_state['df_raw'] is None:
    st.info("👋 Silakan pilih sumber data di sidebar untuk memulai.")
    st.stop()

if st.session_state['df_daily'] is None:
    st.error("❌ Error dalam memproses data. Pastikan format CSV sesuai.")
    st.stop()

df_daily = st.session_state['df_daily']

# Prepare Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Forecast & Fase Event", 
    "🔗 Analisis Korelasi", 
    "📅 Pola Musiman", 
    "🧮 Simulasi Skenario", 
    "📊 Visualisasi Data",
    "📋 Export & Detail"
])

with tab1:
    st.subheader(f"🎯 Analisis Siklus Event: {target_col} ({target_year})")
    
    if st.session_state['custom_holidays']:
        st.success(f"✅ **Mode Hybrid Aktif:** {len(st.session_state['custom_holidays'])} jadwal event terdaftar")
    
    start_forecast_btn = st.button("🚀 Mulai Analisis", type="primary", use_container_width=True)

    if start_forecast_btn or st.session_state['forecast_result'] is not None:
        if start_forecast_btn:
            with st.spinner("🔄 Menganalisis pola dan membangun model prediksi..."):
                
                try:
                    # --- INTELLIGENT HOLIDAY GENERATION ---
                    holidays_df = None
                    if st.session_state['custom_holidays']:
                        holiday_records = []
                        for h in st.session_state['custom_holidays']:
                            # Event itself
                            curr_date = h['ds_start']
                            while curr_date <= h['ds_end']:
                                holiday_records.append({
                                    'holiday': h['holiday'],
                                    'ds': pd.to_datetime(curr_date)
                                })
                                curr_date += timedelta(days=1)
                            
                            # Preparation phase
                            start_prep = h['ds_start'] - timedelta(days=prep_days)
                            end_prep = h['ds_start'] - timedelta(days=1)
                            
                            curr_prep = start_prep
                            while curr_prep <= end_prep:
                                holiday_records.append({
                                    'holiday': f"Pra-{h['holiday']}",
                                    'ds': pd.to_datetime(curr_prep)
                                })
                                curr_prep += timedelta(days=1)
                                
                        holidays_df = pd.DataFrame(holiday_records)

                    # 1. Setup Training and Testing Data
                    train_data = df_daily[df_daily.index.year < target_year].copy()
                    
                    if len(train_data) < 60:
                        st.error("❌ Data training terlalu sedikit (min 60 hari). Pilih tahun target yang lebih besar.")
                        st.stop()
                    
                    if is_backtesting:
                        test_data = df_daily[df_daily.index.year == target_year].copy()
                        days_to_predict = len(test_data) if len(test_data) > 0 else 365
                        future_dates = test_data.index
                    else:
                        last_date = train_data.index.max()
                        days_to_predict = 365
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)

                    # 2. Forecast Regressors (if any)
                    future_regressors_df = pd.DataFrame({'ds': future_dates})
                    
                    if selected_regressors:
                        progress_bar = st.progress(0)
                        for idx, reg_col in enumerate(selected_regressors):
                            progress_bar.progress((idx + 1) / (len(selected_regressors) + 1))
                            
                            m_reg = Prophet(
                                yearly_seasonality=True,
                                weekly_seasonality=True,
                                daily_seasonality=False,
                                holidays=holidays_df,
                                **st.session_state['model_params']
                            )
                            
                            # Safe dataframe preparation
                            df_reg = train_data[[reg_col]].reset_index()
                            # Rename columns: first col is index/date, second is value
                            df_reg.rename(columns={df_reg.columns[0]: 'ds', reg_col: 'y'}, inplace=True)
                            
                            m_reg.fit(df_reg)
                            
                            future_reg = m_reg.make_future_dataframe(periods=days_to_predict)
                            fcst_reg = m_reg.predict(future_reg)
                            
                            future_vals = fcst_reg[fcst_reg['ds'].isin(future_dates)]['yhat'].values
                            if len(future_vals) != len(future_regressors_df):
                                future_vals = np.resize(future_vals, len(future_regressors_df))
                            future_regressors_df[reg_col] = future_vals
                        
                        progress_bar.progress(1.0)
                        progress_bar.empty()
                    
                    # 3. Forecast Main Target
                    m_target = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        holidays=holidays_df,
                        **st.session_state['model_params']
                    )
                    
                    for reg in selected_regressors:
                        m_target.add_regressor(reg)
                        
                    # --- ROBUST DATAFRAME PREPARATION (FIX FOR KEYERROR) ---
                    df_target = train_data.reset_index()
                    # Identify the date column (it's typically the first one after reset)
                    date_col_name = df_target.columns[0]
                    # Rename securely
                    df_target = df_target.rename(columns={date_col_name: 'ds', target_col: 'y'})
                    # Select only necessary columns
                    df_target = df_target[['ds', 'y'] + selected_regressors]
                    
                    m_target.fit(df_target)
                    
                    future_target = m_target.make_future_dataframe(periods=days_to_predict)
                    
                    for reg in selected_regressors:
                        hist_map = train_data[reg].to_dict()
                        fut_map = dict(zip(future_regressors_df['ds'], future_regressors_df[reg]))
                        future_target[reg] = future_target['ds'].map(hist_map).fillna(future_target['ds'].map(fut_map))
                    
                    future_target = future_target.fillna(method='ffill').fillna(method='bfill')
                    forecast = m_target.predict(future_target)
                    
                    # Store in session
                    st.session_state['forecast_model'] = m_target
                    st.session_state['forecast_result'] = forecast
                    st.session_state['future_target_df'] = future_target
                    
                    st.success("✅ Analisis selesai!")
                    
                except Exception as e:
                    st.error(f"❌ Error saat forecasting: {str(e)}")
                    st.stop()
        
        # Retrieve from session
        forecast = st.session_state['forecast_result']
        
        # Filter for target year
        if is_backtesting:
            test_data = df_daily[df_daily.index.year == target_year].copy()
            future_dates = test_data.index
        else:
            train_data = df_daily[df_daily.index.year < target_year].copy()
            last_date = train_data.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365)
        
        forecast_year = forecast[forecast['ds'].dt.year == target_year].copy()
        
        # --- CALCULATE METRICS (if backtesting) ---
        if is_backtesting and len(test_data) > 0:
            actual_vals = test_data[target_col].values
            predicted_vals = forecast_year['yhat'].values[:len(actual_vals)]
            
            if len(predicted_vals) == len(actual_vals):
                metrics = calculate_metrics(actual_vals, predicted_vals)
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">MAE</div>
                        <div class="metric-value">{metrics['MAE']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">{metrics['RMSE']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">MAPE</div>
                        <div class="metric-value">{metrics['MAPE']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{metrics['Accuracy']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # --- DETECT EVENT PHASES ---
        event_phases, threshold = detect_event_phases(
            forecast_year, 
            threshold_multiplier=threshold_mult,
            prep_days=prep_days,
            post_days=post_days
        )
        
        # --- VISUALIZE FORECAST WITH PHASES ---
        fig = go.Figure()
        
        # Historical data
        if is_backtesting:
            hist_data = df_daily[df_daily.index.year < target_year]
        else:
            hist_data = df_daily.copy()
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data[target_col],
            mode='lines',
            name='Data Historis',
            line=dict(color='#94a3b8', width=2)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_year['ds'],
            y=forecast_year['yhat'],
            mode='lines',
            name='Prediksi',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_year['ds'],
            y=forecast_year['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_year['ds'],
            y=forecast_year['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Actual data if backtesting
        if is_backtesting and len(test_data) > 0:
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data[target_col],
                mode='markers',
                name='Aktual',
                marker=dict(color='#10b981', size=4)
            ))
        
        # Threshold line
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: {threshold:.0f}",
            annotation_position="right"
        )
        
        # Mark event phases
        for phase in event_phases:
            # Pra-Event
            fig.add_vrect(
                x0=phase['pra_start'], x1=phase['pra_end'],
                fillcolor="orange", opacity=0.15,
                layer="below", line_width=0,
                annotation_text="Pra-Event",
                annotation_position="top left"
            )
            # Peak
            fig.add_vrect(
                x0=phase['peak_start'], x1=phase['peak_end'],
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="PEAK",
                annotation_position="top left"
            )
            # Pasca-Event
            fig.add_vrect(
                x0=phase['pasca_start'], x1=phase['pasca_end'],
                fillcolor="green", opacity=0.15,
                layer="below", line_width=0,
                annotation_text="Pasca-Event",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title=f"Prediksi {target_col} - Tahun {target_year}",
            xaxis_title="Tanggal",
            yaxis_title=target_col,
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- DISPLAY EVENT PHASES ---
        if event_phases:
            st.markdown("### 📅 Fase-Fase Event Terdeteksi")
            
            for i, phase in enumerate(event_phases, 1):
                with st.expander(f"Event #{i}: {phase['peak_start'].strftime('%d %b')} - {phase['peak_end'].strftime('%d %b %Y')}", expanded=True):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="event-card pra-event">
                            <h4>🟡 Fase Persiapan</h4>
                            <p><strong>Periode:</strong><br>{phase['pra_start'].strftime('%d %b')} - {phase['pra_end'].strftime('%d %b %Y')}</p>
                            <p><strong>Durasi:</strong> {(phase['pra_end'] - phase['pra_start']).days + 1} hari</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="event-card peak-event">
                            <h4>🔴 Fase Peak</h4>
                            <p><strong>Periode:</strong><br>{phase['peak_start'].strftime('%d %b')} - {phase['peak_end'].strftime('%d %b %Y')}</p>
                            <p><strong>Peak Value:</strong> {phase['peak_value']:.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="event-card pasca-event">
                            <h4>🟢 Fase Monitoring</h4>
                            <p><strong>Periode:</strong><br>{phase['pasca_start'].strftime('%d %b')} - {phase['pasca_end'].strftime('%d %b %Y')}</p>
                            <p><strong>Durasi:</strong> {(phase['pasca_end'] - phase['pasca_start']).days + 1} hari</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("#### 💡 Rekomendasi Aksi")
                    st.info(f"""
                    **Persiapan ({phase['pra_start'].strftime('%d %b')} - {phase['pra_end'].strftime('%d %b')}):**
                    - Scaling infrastruktur server
                    - Stock monitoring dan supply chain preparation
                    - Team briefing dan resource allocation
                    
                    **Peak Time ({phase['peak_start'].strftime('%d %b')} - {phase['peak_end'].strftime('%d %b')}):**
                    - 24/7 monitoring aktif
                    - Quick response team standby
                    - Real-time dashboard monitoring
                    
                    **Pasca Event ({phase['pasca_start'].strftime('%d %b')} - {phase['pasca_end'].strftime('%d %b')}):**
                    - Post-mortem analysis
                    - Performance metrics evaluation
                    - Lessons learned documentation
                    """)
        else:
            st.info("ℹ️ Tidak ada lonjakan signifikan terdeteksi. Coba turunkan nilai sensitivitas threshold.")

with tab2:
    st.subheader("🔗 Analisis Korelasi Antar Variabel")
    
    # Calculate correlation matrix
    corr_matrix = df_daily.corr()
    
    # Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Matriks Korelasi'
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Insights
    st.markdown("### 🎯 Insight Korelasi")
    
    if target_col in corr_matrix.columns:
        target_corrs = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        
        st.markdown(f"**Variabel yang paling berkorelasi dengan {target_col}:**")
        for var, corr_val in target_corrs.head(3).items():
            emoji = "✅" if corr_val > 0.7 else "⚠️" if corr_val > 0.4 else "❌"
            st.write(f"{emoji} **{var}**: {corr_val:.3f} ({'Kuat' if abs(corr_val) > 0.7 else 'Sedang' if abs(corr_val) > 0.4 else 'Lemah'})")
        
        st.info("""
        **Interpretasi:**
        - Korelasi > 0.7: Hubungan sangat kuat (gunakan sebagai prediktor)
        - Korelasi 0.4-0.7: Hubungan sedang (pertimbangkan sebagai prediktor)
        - Korelasi < 0.4: Hubungan lemah (tidak disarankan sebagai prediktor)
        """)

with tab3:
    st.subheader("📅 Analisis Pola Musiman")
    
    if st.session_state['forecast_model'] is not None:
        m = st.session_state['forecast_model']
        
        # Prophet's built-in plot
        from prophet.plot import plot_components_plotly
        
        future_df = st.session_state.get('future_target_df')
        if future_df is not None:
            fig_comp = plot_components_plotly(m, st.session_state['forecast_result'])
            st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("""
        **Interpretasi Komponen:**
        - **Trend**: Menunjukkan arah pertumbuhan/penurunan jangka panjang
        - **Weekly**: Pola mingguan (hari apa yang paling tinggi/rendah)
        - **Yearly**: Pola tahunan (bulan apa yang paling tinggi/rendah)
        """)
    else:
        st.info("ℹ️ Jalankan forecasting terlebih dahulu di Tab 'Forecast & Fase Event'")

with tab4:
    st.subheader("🧮 Simulasi Skenario What-If")
    
    if st.session_state['forecast_result'] is not None and selected_regressors:
        st.info("""
        **Gunakan simulasi ini untuk:**
        - Memprediksi dampak perubahan variabel input
        - Testing berbagai skenario bisnis
        - Perencanaan kapasitas
        """)
        
        scenario_adjustments = {}
        cols = st.columns(len(selected_regressors))
        
        for idx, reg in enumerate(selected_regressors):
            with cols[idx]:
                adjustment = st.slider(
                    f"Adjust {reg}",
                    -50, 50, 0, 5,
                    help=f"Perubahan {reg} dalam persen"
                )
                scenario_adjustments[reg] = 1 + (adjustment / 100)
        
        if st.button("🔄 Jalankan Simulasi", use_container_width=True):
            with st.spinner("Menghitung skenario..."):
                future_scenario = st.session_state['future_target_df'].copy()
                
                for reg, factor in scenario_adjustments.items():
                    if reg in future_scenario.columns:
                        future_scenario[reg] = future_scenario[reg] * factor
                
                forecast_scenario = st.session_state['forecast_model'].predict(future_scenario)
                forecast_scenario_year = forecast_scenario[forecast_scenario['ds'].dt.year == target_year]
                
                # Compare with original
                forecast_original_year = st.session_state['forecast_result'][
                    st.session_state['forecast_result']['ds'].dt.year == target_year
                ]
                
                fig_scenario = go.Figure()
                
                fig_scenario.add_trace(go.Scatter(
                    x=forecast_original_year['ds'],
                    y=forecast_original_year['yhat'],
                    mode='lines',
                    name='Prediksi Original',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                fig_scenario.add_trace(go.Scatter(
                    x=forecast_scenario_year['ds'],
                    y=forecast_scenario_year['yhat'],
                    mode='lines',
                    name='Prediksi Skenario',
                    line=dict(color='#f59e0b', width=2, dash='dash')
                ))
                
                fig_scenario.update_layout(
                    title="Perbandingan: Original vs Skenario",
                    xaxis_title="Tanggal",
                    yaxis_title=target_col,
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_scenario, use_container_width=True)
                
                # Calculate impact
                original_sum = forecast_original_year['yhat'].sum()
                scenario_sum = forecast_scenario_year['yhat'].sum()
                diff_pct = ((scenario_sum - original_sum) / original_sum) * 100
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Original Total", f"{original_sum:,.0f}")
                with col_s2:
                    st.metric("Skenario Total", f"{scenario_sum:,.0f}")
                with col_s3:
                    st.metric("Perubahan", f"{diff_pct:+.1f}%")
    else:
        st.info("ℹ️ Simulasi skenario memerlukan prediktor. Pilih minimal 1 prediktor di sidebar.")

with tab5:
    st.subheader("📊 Visualisasi Tren Data")
    
    viz_cols = st.multiselect(
        "Pilih variabel untuk divisualisasi:",
        df_daily.columns.tolist(),
        default=[target_col]
    )
    
    if viz_cols:
        viz_type = st.radio("Tipe Visualisasi:", ["Line Chart", "Area Chart", "Bar Chart"], horizontal=True)
        
        fig_viz = go.Figure()
        
        for col in viz_cols:
            if viz_type == "Line Chart":
                fig_viz.add_trace(go.Scatter(
                    x=df_daily.index, y=df_daily[col],
                    mode='lines', name=col
                ))
            elif viz_type == "Area Chart":
                fig_viz.add_trace(go.Scatter(
                    x=df_daily.index, y=df_daily[col],
                    mode='lines', name=col, fill='tozeroy'
                ))
            else:
                fig_viz.add_trace(go.Bar(
                    x=df_daily.index, y=df_daily[col], name=col
                ))
        
        fig_viz.update_layout(
            title="Tren Data Historis",
            xaxis_title="Tanggal",
            yaxis_title="Nilai",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_viz, use_container_width=True)
        
        # Statistics
        st.markdown("### 📈 Statistik Deskriptif")
        st.dataframe(df_daily[viz_cols].describe(), use_container_width=True)

with tab6:
    st.subheader("📋 Export & Detail Data")
    
    if st.session_state['forecast_result'] is not None:
        forecast_export = st.session_state['forecast_result'][
            st.session_state['forecast_result']['ds'].dt.year == target_year
        ].copy()
        
        # Preview
        st.markdown("### 👀 Preview Data Forecast")
        st.dataframe(forecast_export[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10))
        
        # Export buttons
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            # CSV Export
            csv_data = forecast_export[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"forecast_{target_col}_{target_year}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_e2:
            # Excel Export with phases
            if event_phases:
                excel_data = export_forecast_results(
                    forecast_export[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                    event_phases, 
                    target_col
                )
                st.download_button(
                    label="📥 Download Excel (dengan Fase)",
                    data=excel_data,
                    file_name=f"forecast_analysis_{target_col}_{target_year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Raw data table
        with st.expander("🔍 Lihat Semua Data", expanded=False):
            st.dataframe(
                forecast_export[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                use_container_width=True,
                height=400
            )
    else:
        st.info("ℹ️ Belum ada data forecast. Jalankan analisis terlebih dahulu.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p><strong>Cerebrum Forecasting System</strong></p>
    <p>Powered by Prophet ML • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)