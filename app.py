import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Cerebrum Forecasting",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
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
    }
    .pra-event { border-left-color: #f59e0b; background-color: #fffbeb; }
    .peak-event { border-left-color: #ef4444; background-color: #fef2f2; }
    .pasca-event { border-left-color: #10b981; background-color: #ecfdf5; }
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
        # Create spike mask
        indices = np.where((date_range >= evt - timedelta(days=2)) & (date_range <= evt + timedelta(days=2)))[0]
        user_counts[indices] *= 2.0
        activity_counts[indices] *= 3.5  # Attempts spike harder
        trans_counts[indices] *= 2.5     # Transactions spike too
        
    records = []
    # Sample down for speed and format
    sample_idx = np.random.choice(n, size=int(n*0.3), replace=False)
    sample_idx.sort()
    
    for i in sample_idx:
        dt = date_range[i]
        
        # Add User records
        for _ in range(int(user_counts[i] / 5)): # Scale down
            records.append([dt, 'User'])
            
        # Add Activity records
        for _ in range(int(activity_counts[i] / 5)):
            records.append([dt, 'Aktivitas'])
            
        # Add Transaction records
        for _ in range(int(trans_counts[i] / 5)):
            records.append([dt, 'Transaksi'])
            
    return pd.DataFrame(records, columns=['created_at', 'kategori'])

@st.cache_data(show_spinner=False)
def load_local_csv(file_path):
    """Load large CSV from local path with caching"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading local file: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def process_multivariate_data(df):
    """
    Pivot data to create columns for each category per day.
    This enables multivariate regression.
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    
    df['date'] = df['created_at'].dt.date
    
    # Pivot: Index=Date, Columns=Kategori, Values=Count
    daily_pivot = df.groupby(['date', 'kategori']).size().unstack(fill_value=0)
    daily_pivot.index = pd.to_datetime(daily_pivot.index)
    
    # Add Total column
    daily_pivot['Total'] = daily_pivot.sum(axis=1)
    
    return daily_pivot

def detect_event_phases(forecast_df, threshold_multiplier=1.5, prep_days=7, post_days=7):
    """
    Identifies Pra-Event, Peak, and Pasca-Event phases based on forecast.
    """
    df = forecast_df.copy()
    mean_val = df['yhat'].mean()
    std_val = df['yhat'].std()
    threshold = mean_val + (threshold_multiplier * std_val)
    
    # Identify Peak Days
    df['is_peak'] = df['yhat'] > threshold
    
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
            'pasca_end': pasca_end,
            'peak_value': row['peak_val']
        })
        
    return event_phases, threshold

# --- Session State Initialization ---
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_daily' not in st.session_state:
    st.session_state['df_daily'] = None
if 'data_source_mode' not in st.session_state:
    st.session_state['data_source_mode'] = None
if 'custom_holidays' not in st.session_state:
    st.session_state['custom_holidays'] = []

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
        st.session_state['df_raw'] = None # Clear data on mode change
    
    if data_source == "☁️ Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV (w/ 'kategori' col)", type=['csv'])
        if uploaded_file:
            st.session_state['df_raw'] = pd.read_csv(uploaded_file)
            
    elif data_source == "🖥️ File Lokal (>200MB)":
        st.info("ℹ️ Mode ini membaca file langsung dari folder server untuk bypass limit browser.")
        default_path = "folder/dataset.csv"
        local_path = st.text_input("Path File CSV:", default_path)
        
        # Use session state to persist loaded data across reruns
        if st.button("Muat Data Lokal"):
            if os.path.exists(local_path):
                with st.spinner("Memuat data besar..."):
                    st.session_state['df_raw'] = load_local_csv(local_path)
            else:
                st.warning(f"⚠️ File tidak ditemukan di: {local_path}")
        
        # Show status if data is already loaded in session
        if st.session_state['df_raw'] is not None:
             st.success("✅ Data Lokal Tersimpan di Memori")

    else: # Demo
        if st.session_state['df_raw'] is None:
            st.session_state['df_raw'] = generate_dummy_data()

    # --- PROCESS DATA IF AVAILABLE IN SESSION STATE ---
    if st.session_state['df_raw'] is not None:
        # Process Data (only if not already processed or needs update)
        with st.spinner("Memproses struktur data..."):
            df_daily = process_multivariate_data(st.session_state['df_raw'])
            st.session_state['df_daily'] = df_daily
        
        st.success(f"✅ Data Loaded: {len(df_daily)} days")
        st.markdown("---")
        
        st.header("🎯 Forecasting Strategy")
        
        # --- PANDUAN PEMILIHAN VARIABEL ---
        with st.expander("📚 Panduan: Apa yang harus saya pilih?", expanded=False):
            st.info("""
            **1. Target Utama (Output):**
            Pilih metrik yang ingin Anda prediksi masa depannya.
            - Pilih **'Transaksi'** untuk revenue/omset.
            - Pilih **'Aktivitas'** untuk beban server/traffic.
            - Pilih **'User'** untuk pertumbuhan user baru.
            
            **2. Prediktor (Input):**
            Pilih variabel yang **mempengaruhi** target tersebut.
            - Jika Target = **'Transaksi'**, Prediktor yang bagus biasanya **'User'** dan **'Aktivitas'** (karena user aktif cenderung bertransaksi).
            - Jika Target = **'Total'**, kosongkan prediktor (Univariate).
            """)
        # ----------------------------------
        
        available_cols = list(df_daily.columns)
        
        # 1. Select Target
        target_col = st.selectbox(
            "Target Utama:",
            available_cols,
            index=available_cols.index('Total') if 'Total' in available_cols else 0
        )
        
        # 2. Select Regressors
        potential_regressors = [c for c in available_cols if c != target_col and c != 'Total']
        selected_regressors = st.multiselect(
            "Gunakan sebagai Prediktor:",
            potential_regressors,
            default=[c for c in potential_regressors if c in ['User', 'Aktivitas']] if target_col == 'Transaksi' else []
        )
        
        st.markdown("---")
        
        # 3. Year Selection
        min_year = df_daily.index.year.min()
        max_year = df_daily.index.year.max()
        target_year = st.selectbox("Tahun Target:", range(min_year, max_year + 2), index=len(range(min_year, max_year + 2))-1)
        
        is_backtesting = target_year <= max_year

        # 4. Event Settings
        with st.expander("⚙️ Konfigurasi Deteksi Event", expanded=True):
            threshold_mult = st.slider("Sensitivitas Peak (StdDev)", 1.0, 3.0, 1.5, 0.1)
            prep_days = st.slider("Durasi Pra-Event (Hari)", 1, 14, 14, help="Berapa hari 'ramai' sebelum event dimulai?")
            post_days = st.slider("Durasi Pasca-Event (Hari)", 1, 14, 7, help="Berapa hari monitoring setelah event?")

        # 5. MANUAL HOLIDAY INPUT (NEW FEATURE)
        with st.expander("📅 Input Jadwal Event Resmi (Wajib)", expanded=True):
            st.warning("""
            **PENTING:** Agar grafik **NAIK SEBELUM EVENT**, masukkan jadwal di sini.
            Sistem akan otomatis menandai **14 hari sebelum tanggal ini** sebagai masa "Persiapan Intensif".
            """)
            
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                h_name = st.text_input("Nama Event (misal: CPNS)", "Event CPNS")
            with col_h2:
                h_dates = st.date_input("Rentang Tanggal Event (Hari H)", [])
            
            if st.button("➕ Tambah Jadwal"):
                if len(h_dates) == 2:
                    st.session_state['custom_holidays'].append({
                        'holiday': h_name,
                        'ds_start': h_dates[0],
                        'ds_end': h_dates[1]
                    })
                    st.success(f"Jadwal '{h_name}' ditambahkan! Pra-Event {prep_days} hari akan otomatis dihitung.")
                else:
                    st.error("Mohon pilih Rentang Tanggal (Start & End)")
            
            # Show added holidays
            if st.session_state['custom_holidays']:
                st.markdown("##### Jadwal Tersimpan:")
                holidays_df_display = pd.DataFrame(st.session_state['custom_holidays'])
                st.dataframe(holidays_df_display, use_container_width=True, hide_index=True)
                
                if st.button("🗑️ Reset Jadwal"):
                    st.session_state['custom_holidays'] = []
                    st.rerun()

# --- Main Logic ---

if st.session_state['df_raw'] is None:
    st.info("👋 Silakan pilih sumber data di sidebar dan klik 'Muat Data' untuk memulai.")
    st.stop()

# Use the processed data from session/scope
df_daily = st.session_state['df_daily']

# Prepare Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Forecast & Fase Event", "🔗 Analisis Korelasi", "📅 Pola Musiman", "🧮 Simulasi Skenario", "📋 Data Detail"])

# Global vars for shared data across tabs
if 'forecast_model' not in st.session_state:
    st.session_state['forecast_model'] = None
if 'forecast_result' not in st.session_state:
    st.session_state['forecast_result'] = None
if 'future_target_df' not in st.session_state:
    st.session_state['future_target_df'] = None

with tab1:
    st.subheader(f"Analisis Siklus Event: {target_col} ({target_year})")
    
    if st.session_state['custom_holidays']:
        st.info(f"💡 **Mode Hybrid Aktif:** Model otomatis memperhitungkan **Masa Persiapan {prep_days} Hari** sebelum jadwal resmi yang Anda input.")
    
    start_forecast_btn = st.button("🚀 Mulai Analisis", type="primary")

    if start_forecast_btn or st.session_state['forecast_result'] is not None:
        if start_forecast_btn:
            with st.spinner("Menganalisis pola Pra-Event dan Peak Time..."):
                
                # --- INTELLIGENT HOLIDAY GENERATION ---
                # Key fix: Automatically generate "Pra-Event" days to force the model 
                # to recognize the ramp-up period BEFORE the actual event.
                holidays_df = None
                if st.session_state['custom_holidays']:
                    holiday_records = []
                    for h in st.session_state['custom_holidays']:
                        # 1. The Event Itself (Peak)
                        curr_date = h['ds_start']
                        while curr_date <= h['ds_end']:
                            holiday_records.append({
                                'holiday': h['holiday'],
                                'ds': pd.to_datetime(curr_date)
                            })
                            curr_date += timedelta(days=1)
                        
                        # 2. The Preparation Phase (Ramp Up) - CRITICAL FIX
                        # We force the model to treat the N days before event as "Pra-{EventName}"
                        # This tells Prophet: "Hey, these days are also special/busy!"
                        start_prep = h['ds_start'] - timedelta(days=prep_days)
                        end_prep = h['ds_start'] - timedelta(days=1)
                        
                        curr_prep = start_prep
                        while curr_prep <= end_prep:
                            holiday_records.append({
                                'holiday': f"Pra-{h['holiday']}", # Different name to allow different coefficient
                                'ds': pd.to_datetime(curr_prep)
                            })
                            curr_prep += timedelta(days=1)
                            
                    holidays_df = pd.DataFrame(holiday_records)

                # 1. Setup Data Training vs Testing
                train_data = df_daily[df_daily.index.year < target_year].copy()
                
                if is_backtesting:
                    test_data = df_daily[df_daily.index.year == target_year].copy()
                    days_to_predict = len(test_data) if len(test_data) > 0 else 365
                    future_dates = test_data.index
                else:
                    last_date = train_data.index.max()
                    days_to_predict = 365 # Default 1 year ahead
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)

                # 2. Forecast Regressors (Cascading)
                future_regressors_df = pd.DataFrame({'ds': future_dates})
                
                if selected_regressors:
                    for reg_col in selected_regressors:
                        # Add holidays to regressors too if needed
                        m_reg = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=holidays_df)
                        df_reg = train_data[[reg_col]].reset_index()
                        df_reg.columns = ['ds', 'y']
                        m_reg.fit(df_reg)
                        
                        future_reg = m_reg.make_future_dataframe(periods=days_to_predict)
                        fcst_reg = m_reg.predict(future_reg)
                        
                        future_vals = fcst_reg[fcst_reg['ds'].isin(future_dates)]['yhat'].values
                        if len(future_vals) != len(future_regressors_df):
                            future_vals = np.resize(future_vals, len(future_regressors_df))
                        future_regressors_df[reg_col] = future_vals
                        train_data[reg_col] = train_data[reg_col] 
                
                # 3. Forecast Main Target
                m_target = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=holidays_df)
                for reg in selected_regressors:
                    m_target.add_regressor(reg)
                    
                df_target = train_data.reset_index()[['date', target_col] + selected_regressors]
                df_target.columns = ['ds', 'y'] + selected_regressors
                m_target.fit(df_target)
                
                future_target = m_target.make_future_dataframe(periods=days_to_predict)
                
                for reg in selected_regressors:
                    hist_map = train_data[reg].to_dict()
                    fut_map = dict(zip(future_regressors_df['ds'], future_regressors_df[reg]))
                    future_target[reg] = future_target['ds'].map(hist_map).fillna(future_target['ds'].map(fut_map))
                
                future_target = future_target.fillna(method='ffill').fillna(method='bfill')
                forecast = m_target.predict(future_target)
                
                # Store in session state for other tabs
                st.session_state['forecast_model'] = m_target
                st.session_state['forecast_result'] = forecast
                st.session_state['future_target_df'] = future_target
        
        # Retrieve from state
        forecast = st.session_state['forecast_result']
        
        # Filter for target year (logic repeat for display)
        if is_backtesting:
             test_data = df_daily[df_daily.index.year == target_year].copy()
             future_dates = test_data.index
        else:
             train_data = df_daily[df_daily.index.year < target_year].copy()
             last_date = train_data.index.max()
             future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365)
             
        forecast_only = forecast[forecast['ds'].isin(future_dates)].copy()

        # 4. Detect Events Phase
        phases, threshold = detect_event_phases(forecast_only, threshold_mult, prep_days, post_days)

        # 5. Visualization with Phases
        fig = go.Figure()
        
        # Forecast Line
        fig.add_trace(go.Scatter(
            x=forecast_only['ds'], y=forecast_only['yhat'],
            mode='lines', name=f'Prediksi {target_col}',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # Threshold Line
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray", annotation_text="Peak Threshold")
        
        # Show Manual Holidays on Graph
        if st.session_state['custom_holidays']:
            for h in st.session_state['custom_holidays']:
                # Draw vertical band for manual schedule
                # Ensure date objects are comparable
                start_h = pd.to_datetime(h['ds_start'])
                end_h = pd.to_datetime(h['ds_end'])
                
                # Start of prep
                start_prep = start_h - timedelta(days=prep_days)
                
                # Only draw if within current view range
                if start_h.year == target_year or end_h.year == target_year:
                    # Draw Event
                    fig.add_vrect(
                        x0=start_h, x1=end_h,
                        fillcolor="rgba(147, 51, 234, 0.1)", layer="below", line_width=1, line_dash="dot",
                        annotation_text=f"Event: {h['holiday']}", annotation_position="top left"
                    )
                    # Draw Auto-Generated Prep Phase
                    fig.add_vrect(
                        x0=start_prep, x1=start_h,
                        fillcolor="rgba(245, 158, 11, 0.1)", layer="below", line_width=0,
                        annotation_text=f"Masa Persiapan ({prep_days} hari)", annotation_position="top left"
                    )

        # Add Phase Rectangles
        for p in phases:
            # Pra Event (Yellow)
            fig.add_vrect(
                x0=p['pra_start'], x1=p['pra_end'],
                fillcolor="rgba(245, 158, 11, 0.2)", layer="below", line_width=0,
                annotation_text="Pra", annotation_position="top left"
            )
            # Peak (Red)
            fig.add_vrect(
                x0=p['peak_start'], x1=p['peak_end'],
                fillcolor="rgba(239, 68, 68, 0.2)", layer="below", line_width=0,
                annotation_text="Peak", annotation_position="top left"
            )
            # Pasca (Green)
            fig.add_vrect(
                x0=p['pasca_start'], x1=p['pasca_end'],
                fillcolor="rgba(16, 185, 129, 0.2)", layer="below", line_width=0,
                annotation_text="Pasca", annotation_position="top right"
            )

        fig.update_layout(
            title=f"Timeline Event & Zona Persiapan ({target_year})",
            height=450,
            hovermode='x unified',
            xaxis_title="Tanggal",
            yaxis_title="Volume",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 6. Strategy Cards (The Requested Feature)
        st.markdown("### 🚨 Kartu Strategi & Jadwal Persiapan")
        
        if not phases:
            st.info("Tidak ada event besar terdeteksi dengan sensitivitas saat ini. Coba turunkan nilai Threshold di sidebar.")
        
        for idx, p in enumerate(phases):
            peak_dur = (p['peak_end'] - p['peak_start']).days + 1
            
            with st.expander(f"🗓️ Event #{idx+1}: {p['peak_start'].strftime('%d %b')} - {p['peak_end'].strftime('%d %b %Y')} ({peak_dur} Hari)", expanded=True):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"""
                    <div class='event-card pra-event'>
                        <h4>🟡 FASE PERSIAPAN (PRA)</h4>
                        <p style='font-size: 0.9em; margin-bottom: 5px;'><strong>Mulai:</strong> {p['pra_start'].strftime('%d %b %Y')}</p>
                        <p style='font-size: 0.9em;'><strong>H-{prep_days} Sebelum Peak</strong></p>
                        <hr style='margin: 8px 0;'>
                        <ul style='font-size: 0.85em; padding-left: 15px; margin-bottom: 0;'>
                            <li>Cek kestabilan Server</li>
                            <li>Briefing Tim CS/Teknis</li>
                            <li>Finalisasi Campaign</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div class='event-card peak-event'>
                        <h4>🔴 FASE PUNCAK (PEAK)</h4>
                        <p style='font-size: 0.9em; margin-bottom: 5px;'><strong>Estimasi:</strong> {p['peak_value']:.0f} {target_col}/hari</p>
                        <p style='font-size: 0.9em;'><strong>Durasi:</strong> {peak_dur} Hari</p>
                        <hr style='margin: 8px 0;'>
                        <ul style='font-size: 0.85em; padding-left: 15px; margin-bottom: 0;'>
                            <li>Monitoring Real-time</li>
                            <li>Freeze Code/Deployment</li>
                            <li>Standby Support 24/7</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with c3:
                    st.markdown(f"""
                    <div class='event-card pasca-event'>
                        <h4>🟢 FASE EVALUASI (PASCA)</h4>
                        <p style='font-size: 0.9em; margin-bottom: 5px;'><strong>Selesai:</strong> {p['pasca_end'].strftime('%d %b %Y')}</p>
                        <p style='font-size: 0.9em;'><strong>H+{post_days} Setelah Peak</strong></p>
                        <hr style='margin: 8px 0;'>
                        <ul style='font-size: 0.85em; padding-left: 15px; margin-bottom: 0;'>
                            <li>Analisis Data Log</li>
                            <li>Restrospektif Tim</li>
                            <li>Normalisasi Server</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        # WMAPE Accuracy Display
        if is_backtesting and 'test_data' in locals():
            merged = pd.merge(test_data.reset_index(), forecast_only[['ds', 'yhat']], left_on='date', right_on='ds')
            sum_actuals = merged[target_col].sum()
            sum_errors = np.sum(np.abs(merged[target_col] - merged['yhat']))
            wmape = (sum_errors / sum_actuals) * 100 if sum_actuals > 0 else 0
            accuracy_score = max(0, 100 - wmape)
            
            st.markdown("---")
            st.markdown(f"**Akurasi Model (WMAPE):** `{accuracy_score:.1f}%` (Berdasarkan data aktual tahun {target_year})")

with tab2:
    st.header("🔗 Analisis Korelasi (Hubungan Antar Variabel)")
    corr = df_daily.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", range_color=[-1, 1])
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("📅 Pola Musiman (Decomposition)")
    if st.session_state['forecast_result'] is not None:
        forecast = st.session_state['forecast_result']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Tren Tahunan (Yearly)")
            if 'yearly' in forecast.columns:
                fig_yearly = go.Figure()
                # Plot yearly component
                # Extract first 365 days to show one cycle
                cycle = forecast.iloc[:365].copy()
                cycle['day_of_year'] = cycle['ds'].dt.dayofyear
                cycle = cycle.sort_values('day_of_year')
                
                fig_yearly.add_trace(go.Scatter(x=cycle['ds'].dt.strftime('%b'), y=cycle['yearly'], mode='lines', line=dict(color='purple', width=2)))
                fig_yearly.update_layout(title="Kapan bulan tersibuk?", xaxis_title="Bulan", yaxis_title="Impact")
                st.plotly_chart(fig_yearly, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk pola tahunan.")

        with col2:
            st.subheader("📆 Tren Mingguan (Weekly)")
            if 'weekly' in forecast.columns:
                # Group by day name
                forecast['day_name'] = forecast['ds'].dt.day_name()
                weekly_avg = forecast.groupby('day_name')['weekly'].mean().reindex(
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                )
                
                fig_weekly = px.bar(x=weekly_avg.index, y=weekly_avg.values, 
                                   labels={'x': 'Hari', 'y': 'Impact'},
                                   color=weekly_avg.values, color_continuous_scale='Teal')
                fig_weekly.update_layout(title="Kapan hari tersibuk?")
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk pola mingguan.")
                
        st.subheader("📈 Tren Jangka Panjang")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], line=dict(color='gray', width=2)))
        fig_trend.update_layout(title="Arah Pertumbuhan Bisnis (Trend)", xaxis_title="Tahun", yaxis_title="Base Value")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    else:
        st.warning("⚠️ Silakan jalankan analisis di Tab 1 terlebih dahulu.")

with tab4:
    st.header("🧮 Simulasi Skenario (What-If)")
    st.markdown("Ubah variabel prediktor untuk melihat dampaknya terhadap masa depan.")
    
    if st.session_state['forecast_model'] is not None and selected_regressors:
        st.info("💡 Geser slider di bawah untuk mensimulasikan kenaikan/penurunan metrik pendukung.")
        
        m_target = st.session_state['forecast_model']
        base_future = st.session_state['future_target_df'].copy()
        
        # Slicers for simulation
        multipliers = {}
        cols = st.columns(len(selected_regressors))
        
        for idx, reg in enumerate(selected_regressors):
            with cols[idx]:
                pct = st.slider(f"Perubahan {reg} (%)", -50, 50, 0, 5, key=f"sim_{reg}")
                multipliers[reg] = 1 + (pct / 100)
        
        # Apply multipliers to future data ONLY (dates > last training date)
        # Identify future rows logic
        sim_future = base_future.copy()
        # Find split point
        # Assuming all rows in future_target_df are relevant for prediction
        # We modify the regressor columns
        
        for reg, mult in multipliers.items():
            sim_future[reg] = sim_future[reg] * mult
            
        # Predict again
        sim_forecast = m_target.predict(sim_future)
        
        # Compare visual
        fig_sim = go.Figure()
        
        # Filter for display (Target Year only)
        display_mask = sim_forecast['ds'].dt.year == target_year
        base_display = st.session_state['forecast_result'][display_mask]
        sim_display = sim_forecast[display_mask]
        
        # Base Line
        fig_sim.add_trace(go.Scatter(
            x=base_display['ds'], y=base_display['yhat'],
            mode='lines', name='Baseline (Awal)',
            line=dict(color='gray', dash='dash')
        ))
        
        # Simulated Line
        fig_sim.add_trace(go.Scatter(
            x=sim_display['ds'], y=sim_display['yhat'],
            mode='lines', name='Skenario Simulasi',
            line=dict(color='#2563eb', width=3)
        ))
        
        # Calculate Difference
        total_base = base_display['yhat'].sum()
        total_sim = sim_display['yhat'].sum()
        diff_pct = ((total_sim - total_base) / total_base) * 100
        
        st.metric(f"Dampak ke {target_col}", f"{diff_pct:+.2f}%", help="Perubahan total volume setahun")
        
        fig_sim.update_layout(title=f"Perbandingan Baseline vs Simulasi ({target_year})", hovermode='x unified')
        st.plotly_chart(fig_sim, use_container_width=True)
        
    elif not selected_regressors:
        st.warning("⚠️ Fitur simulasi hanya aktif jika Anda menggunakan Multivariate (memilih Prediktor).")
    else:
        st.warning("⚠️ Silakan jalankan analisis di Tab 1 terlebih dahulu.")

with tab5:
    st.header("📋 Data Detail")
    st.dataframe(df_daily.sort_index(ascending=False), use_container_width=True)
    csv = df_daily.to_csv().encode('utf-8')
    st.download_button("📥 Download CSV", csv, "forecast_data.csv", "text/csv")