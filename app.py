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
from itertools import product
import mysql.connector # Library untuk koneksi MySQL
import urllib.parse # Library untuk format URL Google Calendar

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
        margin-bottom: 0.5rem;
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

# --- GOOGLE CALENDAR HELPER ---
def create_gcal_link(title, start_date, end_date, description, location="Online/On-Site"):
    """
    Generate Google Calendar Link
    Note: For all-day events in GCal, end date is exclusive (must be +1 day)
    """
    # Format: YYYYMMDD
    start_str = start_date.strftime('%Y%m%d')
    # Add 1 day to end date because GCal All-Day event end date is exclusive
    end_dt_obj = end_date + timedelta(days=1)
    end_str = end_dt_obj.strftime('%Y%m%d')
    
    base_url = "https://www.google.com/calendar/render?action=TEMPLATE"
    params = {
        "text": title,
        "dates": f"{start_str}/{end_str}",
        "details": description,
        "location": location
    }
    query_string = urllib.parse.urlencode(params)
    return f"{base_url}&{query_string}"

# --- LOGGING FUNCTIONALITY ---
# CONSTANT: Database khusus untuk menyimpan log
LOG_DB_NAME = 'cerebrum_forecast_logs'

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

# --- DATABASE FUNCTIONS ---

def get_db_connection(host, port, user, password, database=None):
    """Generic function to get DB connection"""
    try:
        return mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database
        )
    except mysql.connector.Error as err:
        st.error(f"❌ Koneksi Database Error ({database}): {err}")
        return None

def get_database_list(host, port, user, password):
    """Connect to MySQL and retrieve list of available databases"""
    conn = get_db_connection(host, port, user, password)
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in cursor.fetchall()]
            conn.close()
            return databases
        except Exception as e:
            st.error(f"Error fetching databases: {e}")
            return []
    return []

@st.cache_data(show_spinner=False, ttl=3600) 
def fetch_data_from_db(host, port, user, password, selected_db):
    """Fetch data dynamically from selected MySQL Database"""
    
    query = f"""
    /* --- 1. DATA USER --- */
    SELECT id, created_at, '{selected_db}' AS source_database, 'User' AS kategori
    FROM {selected_db}.tbl_users
    WHERE role_id = 1 and created_at is not null

    UNION ALL

    /* --- 2. DATA AKTIVITAS --- */
    SELECT id, created_at, '{selected_db}' AS source_database, 'Aktivitas' AS kategori
    FROM {selected_db}.trans_attempts
    where created_at is not null

    UNION ALL

    /* --- 3. DATA TRANSAKSI --- */
    SELECT id, created_at, '{selected_db}' AS source_database, 'Transaksi' AS kategori
    FROM {selected_db}.trans_payment_transactions
    WHERE status = 1 and created_at is not null

    ORDER BY created_at DESC;
    """

    conn = get_db_connection(host, port, user, password, selected_db)
    if conn:
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"❌ Error Executing Query: {e}")
            return None
    return None

# --- 3. EVENT LOG CRUD OPERATIONS (ISOLATED DATABASE) ---

def init_log_table(host, port, user, password):
    """Create event_logs table in the dedicated log database if not exists"""
    # Note: Connecting directly to LOG_DB_NAME
    conn = get_db_connection(host, port, user, password, LOG_DB_NAME)
    
    if conn is None:
        st.error(f"❌ Gagal koneksi ke database log '{LOG_DB_NAME}'. Pastikan database ini sudah dibuat di MySQL.")
        return False
        
    try:
        cursor = conn.cursor()
        query = """
        CREATE TABLE IF NOT EXISTS event_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            event_name VARCHAR(255) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            description TEXT,
            source_database VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """
        cursor.execute(query)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Gagal membuat tabel log: {e}")
        return False

def fetch_event_logs_db(host, port, user, password, filter_source_db):
    """Read logs from dedicated Log DB, filtered by source_database"""
    conn = get_db_connection(host, port, user, password, LOG_DB_NAME)
    if conn:
        try:
            # Only fetch logs relevant to the currently analyzed database
            query = "SELECT id, event_name, start_date, end_date, description, source_database, created_at FROM event_logs WHERE source_database = %s ORDER BY start_date DESC"
            df = pd.read_sql(query, conn, params=(filter_source_db,))
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def insert_event_log_db(host, port, user, password, data):
    """Insert new log to dedicated Log DB"""
    conn = get_db_connection(host, port, user, password, LOG_DB_NAME)
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO event_logs (event_name, start_date, end_date, description, source_database) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, data)
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Gagal Insert: {e}")
            return False
    return False

def update_event_log_db(host, port, user, password, log_id, event_name, start_date, end_date, description):
    """Update existing log in dedicated Log DB"""
    conn = get_db_connection(host, port, user, password, LOG_DB_NAME)
    if conn:
        try:
            cursor = conn.cursor()
            query = """
                UPDATE event_logs 
                SET event_name=%s, start_date=%s, end_date=%s, description=%s 
                WHERE id=%s
            """
            cursor.execute(query, (event_name, start_date, end_date, description, int(log_id)))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Gagal Update: {e}")
            return False
    return False

def delete_event_log_db(host, port, user, password, log_id):
    """Delete log from dedicated Log DB"""
    conn = get_db_connection(host, port, user, password, LOG_DB_NAME)
    if conn:
        try:
            cursor = conn.cursor()
            query = "DELETE FROM event_logs WHERE id = %s"
            cursor.execute(query, (int(log_id),))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Gagal Delete: {e}")
            return False
    return False

# --- EXCEL GENERATOR FOR TEMPLATE ---
def generate_excel_template():
    # Menggunakan format dd-mm-yyyy untuk contoh tanggal
    df = pd.DataFrame({
        'Nama Event': ['Promo Lebaran', 'Flash Sale 12.12', 'UTBK - Pendaftaran'],
        'Tanggal Mulai': ['01-03-2025', '12-12-2025', '10-04-2025'], # Format dd-mm-yyyy
        'Tanggal Selesai': ['15-03-2025', '14-12-2025', '20-04-2025'], # Format dd-mm-yyyy
        'Deskripsi': ['Diskon 50%', 'Flash sale akhir tahun', 'Masa pendaftaran']
    })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

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
    # coerce errors to NaT, so we can drop them later
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Remove invalid dates (NaT) immediately
    initial_len = len(df)
    df = df.dropna(subset=['created_at'])
    
    if len(df) == 0:
        st.error("❌ Semua data tanggal tidak valid atau kosong. Periksa format CSV Anda.")
        return None

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
    
    # Validasi Range sebelum reindex
    if daily_pivot.index.empty:
         st.error("❌ Data kosong setelah pemrosesan.")
         return None
         
    min_date = daily_pivot.index.min()
    max_date = daily_pivot.index.max()
    
    # Double check if min/max are valid timestamps (not NaT)
    if pd.isnull(min_date) or pd.isnull(max_date):
        st.error("❌ Terdeteksi tanggal NaT pada index. Cek format data.")
        return None

    # Fill missing dates with 0
    try:
        full_range = pd.date_range(start=min_date, end=max_date, freq='D')
        daily_pivot = daily_pivot.reindex(full_range, fill_value=0)
        # Ensure index name is clear for later use
        daily_pivot.index.name = 'date'
        return daily_pivot
    except Exception as e:
        st.error(f"❌ Error saat membuat date range: {str(e)}")
        return None

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
    
    # Safe MAPE Calculation (avoid division by zero)
    # Kita hanya menghitung MAPE untuk data dimana aktual != 0
    non_zero_mask = actual != 0
    if non_zero_mask.sum() > 0:
        # Menghitung MAPE hanya pada data yang valid (tidak nol)
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = 0.0 # Jika semua data 0, anggap MAPE 0 atau N/A
    
    # WMAPE (Weighted Mean Absolute Percentage Error)
    # Ini lebih aman daripada MAPE untuk data yang banyak nol-nya
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

def auto_tune_prophet(train_df, target_col, regressors, holidays_df, use_log=False):
    """
    Perform Grid Search to find best hyperparameters on training data.
    Uses last 20% of training data as validation set.
    """
    # Split data for tuning
    cutoff = int(len(train_df) * 0.8)
    tune_train = train_df.iloc[:cutoff].copy()
    tune_val = train_df.iloc[cutoff:].copy()
    
    # Prepare tune dataframes
    tune_train_df = tune_train.reset_index()
    tune_train_df = tune_train_df.rename(columns={tune_train_df.columns[0]: 'ds', target_col: 'y'})
    tune_train_df = tune_train_df[['ds', 'y'] + regressors]
    
    tune_val_df = tune_val.reset_index()
    tune_val_df = tune_val_df.rename(columns={tune_val_df.columns[0]: 'ds', target_col: 'y'})
    actuals = tune_val_df['y'].values
    
    if use_log:
        tune_train_df['y'] = np.log1p(tune_train_df['y'])
    
    # Define Parameter Grid
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.5],
        'seasonality_prior_scale': [1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    best_params = None
    best_rmse = float('inf')
    
    # Create future dataframe for validation period
    val_future = pd.DataFrame({'ds': tune_val_df['ds']})
    
    for params in all_params:
        try:
            m = Prophet(
                yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=holidays_df,
                **params
            )
            
            for reg in regressors:
                m.add_regressor(reg)
                
            m.fit(tune_train_df)
            
            # Prepare prediction DF with regressors
            pred_df = val_future.copy()
            for reg in regressors:
                # Naive mapping for tuning speed
                hist_map = train_df[reg].to_dict()
                pred_df[reg] = pred_df['ds'].map(hist_map).fillna(method='ffill')
            
            forecast = m.predict(pred_df)
            
            preds = forecast['yhat'].values
            if use_log:
                preds = np.expm1(preds)
            
            # Clip negative
            preds = np.maximum(preds, 0)
            
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                
        except Exception:
            continue
            
    return best_params

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
if 'auto_tuned_params' not in st.session_state:
    st.session_state['auto_tuned_params'] = None
if 'db_list' not in st.session_state:
    st.session_state['db_list'] = []
if 'selected_db_name' not in st.session_state:
    st.session_state['selected_db_name'] = ""
if 'db_creds' not in st.session_state:
    st.session_state['db_creds'] = {}

# --- Sidebar ---

with st.sidebar:
    st.header("⚙️ Data Configuration")
    
    data_source = st.radio(
        "Sumber Data:", 
        ["📂 Data Demo", "☁️ Upload CSV", "🖥️ File Lokal (>200MB)", "🔌 Database MySQL"]
    )
    
    # Reset data if source changes
    if st.session_state['data_source_mode'] != data_source:
        st.session_state['data_source_mode'] = data_source
        st.session_state['df_raw'] = None
    
    if data_source == "🔌 Database MySQL":
        st.info("ℹ️ Koneksi langsung ke database MySQL.")
        
        with st.expander("🔑 Kredensial Database", expanded=True):
            # Default values as requested
            db_host = st.text_input("Host IP", "34.101.133.82")
            db_port = st.text_input("Port", "3306")
            db_user = st.text_input("Username", "product_intern")
            db_pass = st.text_input("Password", type="password")
            
            if st.button("🔗 Cek Koneksi & Ambil DB"):
                with st.spinner("Menghubungkan..."):
                    dbs = get_database_list(db_host, db_port, db_user, db_pass)
                    if dbs:
                        st.session_state['db_list'] = dbs
                        st.session_state['db_creds'] = {
                            'host': db_host,
                            'port': db_port,
                            'user': db_user,
                            'pass': db_pass
                        }
                        st.success(f"✅ Terhubung! Ditemukan {len(dbs)} database.")
                    else:
                        st.session_state['db_list'] = []
        
        if st.session_state['db_list']:
            selected_db = st.selectbox(
                "Pilih Database:", 
                st.session_state['db_list'],
                index=st.session_state['db_list'].index('jadisekdin_base') if 'jadisekdin_base' in st.session_state['db_list'] else 0
            )
            # Store selected DB for logging
            st.session_state['selected_db_name'] = selected_db
            
            if st.button("📥 Tarik Data"):
                with st.spinner(f"Menarik data dari '{selected_db}'..."):
                    loaded_df = fetch_data_from_db(db_host, db_port, db_user, db_pass, selected_db)
                    if loaded_df is not None and not loaded_df.empty:
                        st.session_state['df_raw'] = loaded_df
                        st.success(f"✅ Berhasil menarik {len(loaded_df)} baris data!")
                    elif loaded_df is not None and loaded_df.empty:
                        st.warning("⚠️ Query berhasil, tetapi data kosong.")
                    else:
                        st.error("❌ Gagal menarik data.")

        if st.session_state['df_raw'] is not None:
            st.success("✅ Data Database Tersimpan di Memori")

    elif data_source == "☁️ Upload CSV":
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
            
            # --- ADVANCED SETTINGS (UPDATED) ---
            with st.expander("⚙️ Optimasi Akurasi Model", expanded=True):
                st.markdown("**🤖 Auto-Tuning**")
                use_auto_tune = st.checkbox(
                    "⚡ Optimasi Parameter Otomatis",
                    value=False,
                    help="Biarkan sistem mencari kombinasi parameter terbaik dengan menguji berbagai skenario pada data training. Proses ini mungkin memakan waktu beberapa detik."
                )
                
                if use_auto_tune and st.session_state['auto_tuned_params']:
                    st.success(f"Best Params: {st.session_state['auto_tuned_params']}")

                st.markdown("---")
                st.markdown("**🔧 Parameter Manual (Diabaikan jika Auto-Tune aktif)**")
                
                # Seasonality Mode
                seasonality_mode = st.selectbox(
                    "Mode Musiman",
                    ["additive", "multiplicative"],
                    index=0,
                    help="'Additive': Fluktuasi tetap (misal: selalu naik 100).\n'Multiplicative': Fluktuasi mengikuti tren (misal: naik 10%). Gunakan ini jika grafik 'melebar' ke kanan.",
                    disabled=use_auto_tune
                )
                
                # Log Transform
                use_log_transform = st.checkbox(
                    "Gunakan Log Transformation",
                    value=False,
                    help="Centang ini jika data Anda memiliki variansi tinggi (misal: selisih nilai min dan max sangat jauh). Ini seringkali MENURUNKAN error secara drastis."
                )
                
                # Outlier Removal
                remove_outliers_pct = st.slider(
                    "Hapus Outlier (Top %)", 
                    0, 10, 0,
                    help="Membuang X% data tertinggi (anomali) dari training data agar model tidak bingung."
                )
                
                col_ft1, col_ft2 = st.columns(2)
                with col_ft1:
                    changepoint_scale = st.slider(
                        "Changepoint Prior",
                        0.001, 0.5, 0.05, 0.001,
                        help="Fleksibilitas trend. Naikkan jika trend sering berubah drastis.",
                        disabled=use_auto_tune
                    )
                with col_ft2:
                    seasonality_scale = st.slider(
                        "Seasonality Prior",
                        0.01, 20.0, 10.0, 0.1,
                        help="Kekuatan pola musiman. Naikkan jika pola harian/bulanan sangat kuat.",
                        disabled=use_auto_tune
                    )
                
                # Logic to set params based on auto-tune or manual
                if not use_auto_tune:
                    st.session_state['model_params'] = {
                        'changepoint_prior_scale': changepoint_scale,
                        'seasonality_prior_scale': seasonality_scale,
                        'seasonality_mode': seasonality_mode
                    }
                    st.session_state['auto_tuned_params'] = None # Reset if switched to manual
                elif st.session_state['auto_tuned_params']:
                    # Apply auto tuned params
                     st.session_state['model_params'] = st.session_state['auto_tuned_params']

            
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

            # 5. MANUAL HOLIDAY INPUT (AUTOMATED FROM DB)
            with st.expander("📅 Jadwal Event Resmi (Aktif)", expanded=True):
                
                # Check for manually added holidays first
                manual_holidays_count = len(st.session_state['custom_holidays'])
                
                # Load from event log if database selected
                log_holidays_count = 0
                if data_source == "🔌 Database MySQL" and st.session_state['selected_db_name']:
                    # Try fetch logs, if fail (e.g. table not exist), it returns empty
                    creds = st.session_state.get('db_creds')
                    if creds:
                         db_events = fetch_event_logs_db(creds['host'], creds['port'], creds['user'], creds['pass'], st.session_state['selected_db_name'])
                         log_holidays_count = len(db_events)
                    
                         if log_holidays_count > 0:
                            st.success(f"✅ {log_holidays_count} event diambil dari Catatan Real ({st.session_state['selected_db_name']})")
                            st.dataframe(db_events[['event_name', 'start_date', 'end_date']], hide_index=True)
                         else:
                            st.info("ℹ️ Belum ada catatan event real di database.")
                
                if manual_holidays_count > 0:
                    st.info(f"📋 {manual_holidays_count} event manual tambahan.")
                    st.dataframe(pd.DataFrame(st.session_state['custom_holidays']), hide_index=True)
                
                if log_holidays_count == 0 and manual_holidays_count == 0:
                    st.warning("⚠️ Tidak ada jadwal event. Prediksi mungkin kurang akurat saat hari raya/promo.")

                # Manual input form still available for ad-hoc additions
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    h_name = st.text_input("Tambah Event Manual", key="holiday_name_manual")
                with col_h2:
                    h_dates = st.date_input("Rentang Tanggal", [], key="holiday_dates_manual")
                
                if st.button("➕ Tambah Manual"):
                    if len(h_dates) == 2:
                        st.session_state['custom_holidays'].append({
                            'holiday': h_name,
                            'ds_start': h_dates[0],
                            'ds_end': h_dates[1]
                        })
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
tabs_list = [
    "🔮 Forecast & Fase Event", 
    "🔗 Analisis Korelasi", 
    "📅 Pola Musiman", 
    "🧮 Simulasi Skenario", 
    "📊 Visualisasi Data", 
    "📋 Export & Detail"
]

# Only show "Catatan Event Real" if connected to Database
if data_source == "🔌 Database MySQL":
    tabs_list.append("📝 Catatan Event Real")

tabs = st.tabs(tabs_list)

# Assign tabs
tab1, tab2, tab3, tab4, tab5, tab6 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5]
tab7 = tabs[6] if len(tabs) > 6 else None

with tab1:
    st.subheader(f"🎯 Analisis Siklus Event: {target_col} ({target_year})")
    
    start_forecast_btn = st.button("🚀 Mulai Analisis", type="primary", use_container_width=True)

    if start_forecast_btn or st.session_state['forecast_result'] is not None:
        if start_forecast_btn:
            with st.spinner("🔄 Menganalisis pola dan membangun model prediksi..."):
                
                try:
                    # --- INTELLIGENT HOLIDAY GENERATION (MERGED) ---
                    holiday_records = []
                    
                    # 1. Add from Manual Input
                    if st.session_state['custom_holidays']:
                        for h in st.session_state['custom_holidays']:
                            # ... (logic same as before, see full code block) ...
                            curr = h['ds_start']
                            while curr <= h['ds_end']:
                                holiday_records.append({'holiday': h['holiday'], 'ds': pd.to_datetime(curr)})
                                curr += timedelta(days=1)
                            # Prep phase logic...
                            start_prep = h['ds_start'] - timedelta(days=prep_days)
                            end_prep = h['ds_start'] - timedelta(days=1)
                            curr = start_prep
                            while curr <= end_prep:
                                holiday_records.append({'holiday': f"Pra-{h['holiday']}", 'ds': pd.to_datetime(curr)})
                                curr += timedelta(days=1)

                    # 2. Add from Event Log (Database Specific)
                    if data_source == "🔌 Database MySQL" and st.session_state['selected_db_name'] and st.session_state.get('db_creds'):
                        creds = st.session_state['db_creds']
                        db_events = fetch_event_logs_db(creds['host'], creds['port'], creds['user'], creds['pass'], st.session_state['selected_db_name'])
                        
                        for _, row in db_events.iterrows():
                            # Parse dates
                            start_dt = pd.to_datetime(row['start_date'])
                            end_dt = pd.to_datetime(row['end_date'])
                            event_name = row['event_name']
                            
                            # Event days
                            curr = start_dt
                            while curr <= end_dt:
                                holiday_records.append({'holiday': event_name, 'ds': curr})
                                curr += timedelta(days=1)
                                
                            # Prep phase
                            start_prep = start_dt - timedelta(days=prep_days)
                            end_prep = start_dt - timedelta(days=1)
                            curr = start_prep
                            while curr <= end_prep:
                                holiday_records.append({'holiday': f"Pra-{event_name}", 'ds': curr})
                                curr += timedelta(days=1)

                    holidays_df = pd.DataFrame(holiday_records) if holiday_records else None

                    # 1. Setup Training and Testing Data
                    train_data = df_daily[df_daily.index.year < target_year].copy()
                    
                    if len(train_data) < 60:
                        st.error("❌ Data training terlalu sedikit (min 60 hari). Pilih tahun target yang lebih besar.")
                        st.stop()
                    
                    # --- FEATURE: OUTLIER REMOVAL ---
                    if remove_outliers_pct > 0:
                        upper_lim = train_data[target_col].quantile(1.0 - (remove_outliers_pct/100.0))
                        original_len = len(train_data)
                        # Filter rows where target column is below the quantile cutoff
                        train_data = train_data[train_data[target_col] <= upper_lim]
                        # Note: We keep rows, not just cap values, to avoid distortion in regressors
                        removed_count = original_len - len(train_data)
                        st.toast(f"🧹 Menghapus {removed_count} outlier dari training data")

                    # --- FEATURE: AUTO-TUNING ---
                    if use_auto_tune:
                         with st.spinner("🤖 Sedang mencari parameter terbaik (Auto-Tuning)..."):
                             best_params = auto_tune_prophet(
                                 train_data, 
                                 target_col, 
                                 selected_regressors, 
                                 holidays_df,
                                 use_log=use_log_transform
                             )
                             st.session_state['auto_tuned_params'] = best_params
                             st.session_state['model_params'] = best_params
                             st.toast(f"✨ Parameter Terbaik: {best_params['seasonality_mode']} | CP: {best_params['changepoint_prior_scale']}")
                    
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
                        
                    # --- ROBUST DATAFRAME PREPARATION ---
                    df_target = train_data.reset_index()
                    date_col_name = df_target.columns[0]
                    df_target = df_target.rename(columns={date_col_name: 'ds', target_col: 'y'})
                    df_target = df_target[['ds', 'y'] + selected_regressors]
                    
                    # --- FEATURE: LOG TRANSFORMATION ---
                    if use_log_transform:
                        df_target['y'] = np.log1p(df_target['y'])
                    
                    m_target.fit(df_target)
                    
                    future_target = m_target.make_future_dataframe(periods=days_to_predict)
                    
                    for reg in selected_regressors:
                        hist_map = train_data[reg].to_dict()
                        fut_map = dict(zip(future_regressors_df['ds'], future_regressors_df[reg]))
                        future_target[reg] = future_target['ds'].map(hist_map).fillna(future_target['ds'].map(fut_map))
                    
                    future_target = future_target.fillna(method='ffill').fillna(method='bfill')
                    forecast = m_target.predict(future_target)
                    
                    # --- INVERSE LOG TRANSFORM (Important!) ---
                    if use_log_transform:
                        forecast['yhat'] = np.expm1(forecast['yhat'])
                        forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
                        forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
                    
                    # Handle negative predictions (common in Prophet, but physically impossible for counts)
                    forecast['yhat'] = forecast['yhat'].clip(lower=0)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                    
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
        for i, phase in enumerate(event_phases, 1):
            # ... existing rectangle drawing logic ...
            fig.add_vrect(x0=phase['pra_start'], x1=phase['pra_end'], fillcolor="orange", opacity=0.15, layer="below", line_width=0, annotation_text="Pra-Event", annotation_position="top left")
            fig.add_vrect(x0=phase['peak_start'], x1=phase['peak_end'], fillcolor="red", opacity=0.2, layer="below", line_width=0, annotation_text="PEAK", annotation_position="top left")
            fig.add_vrect(x0=phase['pasca_start'], x1=phase['pasca_end'], fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="Pasca-Event", annotation_position="top left")
        
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
                        # Add to Calendar Button (Pra-Event)
                        url_pra = create_gcal_link(f"Pra-Event: {target_col}", phase['pra_start'], phase['pra_end'], "Persiapan menghadapi lonjakan trafik.")
                        st.link_button("📅 Add to GCal (Pra)", url_pra)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="event-card peak-event">
                            <h4>🔴 Fase Peak</h4>
                            <p><strong>Periode:</strong><br>{phase['peak_start'].strftime('%d %b')} - {phase['peak_end'].strftime('%d %b %Y')}</p>
                            <p><strong>Peak Value:</strong> {phase['peak_value']:.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        # Add to Calendar Button (Peak)
                        url_peak = create_gcal_link(f"PEAK Event: {target_col}", phase['peak_start'], phase['peak_end'], f"Estimasi Peak Value: {phase['peak_value']:.0f}")
                        st.link_button("📅 Add to GCal (Peak)", url_peak)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="event-card pasca-event">
                            <h4>🟢 Fase Monitoring</h4>
                            <p><strong>Periode:</strong><br>{phase['pasca_start'].strftime('%d %b')} - {phase['pasca_end'].strftime('%d %b %Y')}</p>
                            <p><strong>Durasi:</strong> {(phase['pasca_end'] - phase['pasca_start']).days + 1} hari</p>
                        </div>
                        """, unsafe_allow_html=True)
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

# --- TAB 7: DATABASE LOGGING (NEW & IMPROVED) ---
if data_source == "🔌 Database MySQL" and len(tabs) > 6:
    with tabs[6]:
        st.subheader(f"📝 Database Event Real: {st.session_state.get('selected_db_name', 'Unknown')}")
        
        creds = st.session_state.get('db_creds', {})
        if not creds:
            st.error("Silakan koneksikan database di sidebar terlebih dahulu.")
        else:
            # Check/Create table in isolated LOG DB
            init_log_table(creds['host'], creds['port'], creds['user'], creds['pass'])
            
            with st.expander("📤 Import / Export Jadwal (Excel)", expanded=False):
                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    st.markdown("#### 1. Download Template")
                    st.caption("Gunakan format ini untuk upload data.")
                    template_data = generate_excel_template()
                    st.download_button(
                        label="📥 Download Template Excel",
                        data=template_data,
                        file_name="template_jadwal_event.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col_ex2:
                    st.markdown("#### 2. Upload Jadwal")
                    st.caption("Upload file Excel yang sudah diisi.")
                    uploaded_excel = st.file_uploader("Pilih file Excel (.xlsx)", type=['xlsx'])
                    
                    if uploaded_excel:
                        try:
                            df_import = pd.read_excel(uploaded_excel)
                            # Validate columns
                            required_cols = ['Nama Event', 'Tanggal Mulai', 'Tanggal Selesai']
                            if all(col in df_import.columns for col in required_cols):
                                st.success("✅ Format valid! Preview data:")
                                st.dataframe(df_import.head())
                                
                                if st.button("💾 Simpan Data Import ke Database"):
                                    success_count = 0
                                    for _, row in df_import.iterrows():
                                        # Use default value if Deskripsi missing
                                        desc = row['Deskripsi'] if 'Deskripsi' in row else ''
                                        
                                        # Format dates to ensure they are python date objects or strings YYYY-MM-DD
                                        try:
                                            # Handle format dd-mm-yyyy explicitly
                                            # dayfirst=True membantu pandas mengenali format 01-03-2025 sebagai 1 Maret, bukan 3 Januari
                                            s_date = pd.to_datetime(row['Tanggal Mulai'], dayfirst=True).date()
                                            e_date = pd.to_datetime(row['Tanggal Selesai'], dayfirst=True).date()
                                            
                                            insert_event_log_db(
                                                creds['host'], creds['port'], creds['user'], creds['pass'],
                                                (
                                                    row['Nama Event'],
                                                    s_date,
                                                    e_date,
                                                    desc,
                                                    st.session_state['selected_db_name']
                                                )
                                            )
                                            success_count += 1
                                        except Exception as e:
                                            st.error(f"Gagal import baris {row['Nama Event']}: {e}")
                                    
                                    if success_count > 0:
                                        st.success(f"Berhasil mengimport {success_count} jadwal event!")
                                        st.rerun()
                            else:
                                st.error(f"Format salah. Kolom wajib: {', '.join(required_cols)}")
                        except Exception as e:
                            st.error(f"Error membaca file: {e}")

            # 1. VIEW & EDIT DATA
            st.markdown("##### 📋 Data Event (Editable)")
            st.info("💡 Edit langsung di tabel lalu klik 'Simpan Perubahan'. Klik icon tong sampah untuk menghapus.")
            
            # Load fresh data filtered by current analyzed DB
            df_logs = fetch_event_logs_db(creds['host'], creds['port'], creds['user'], creds['pass'], st.session_state.get('selected_db_name'))
            
            # Prepare dataframe for editor (convert dates to proper type)
            if not df_logs.empty:
                df_logs['start_date'] = pd.to_datetime(df_logs['start_date']).dt.date
                df_logs['end_date'] = pd.to_datetime(df_logs['end_date']).dt.date
            
            # --- EDITOR CONFIG ---
            edited_df = st.data_editor(
                df_logs,
                num_rows="dynamic",
                column_config={
                    "id": st.column_config.NumberColumn(disabled=True),
                    "created_at": st.column_config.DatetimeColumn(disabled=True),
                    "source_database": st.column_config.TextColumn(disabled=True),
                    "start_date": st.column_config.DateColumn("Mulai", format="YYYY-MM-DD"),
                    "end_date": st.column_config.DateColumn("Selesai", format="YYYY-MM-DD"),
                    "event_name": "Nama Event",
                    "description": "Deskripsi"
                },
                key="event_editor"
            )
            
            # --- SYNC BUTTON ---
            if st.button("💾 Simpan Perubahan ke Database"):
                changes = st.session_state["event_editor"]
                
                # A. Handle Added Rows
                for row in changes["added_rows"]:
                    insert_event_log_db(
                        creds['host'], creds['port'], creds['user'], creds['pass'],
                        (
                            row.get('event_name', 'New Event'),
                            row.get('start_date', datetime.now().date()),
                            row.get('end_date', datetime.now().date()),
                            row.get('description', ''),
                            st.session_state['selected_db_name']
                        )
                    )
                
                # B. Handle Deleted Rows
                for idx in changes["deleted_rows"]:
                    # Get the ID of the deleted row from original dataframe
                    # Careful with index if user edits + deletes
                    if idx < len(df_logs):
                        log_id_to_delete = df_logs.iloc[idx]['id']
                        delete_event_log_db(creds['host'], creds['port'], creds['user'], creds['pass'], log_id_to_delete)
                
                # C. Handle Edited Rows
                for idx, row_changes in changes["edited_rows"].items():
                    if idx < len(df_logs):
                        log_id_to_edit = df_logs.iloc[idx]['id']
                        original_row = df_logs.iloc[idx]
                        
                        new_name = row_changes.get('event_name', original_row['event_name'])
                        new_start = row_changes.get('start_date', original_row['start_date'])
                        new_end = row_changes.get('end_date', original_row['end_date'])
                        new_desc = row_changes.get('description', original_row['description'])
                        
                        update_event_log_db(
                            creds['host'], creds['port'], creds['user'], creds['pass'],
                            log_id_to_edit, new_name, new_start, new_end, new_desc
                        )
                
                st.success("✅ Database berhasil diperbarui!")
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p><strong>Cerebrum Forecasting System</strong></p>
    <p>Powered by Prophet ML • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)