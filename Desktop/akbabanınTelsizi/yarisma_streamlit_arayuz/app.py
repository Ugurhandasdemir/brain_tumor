import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import json
import joblib

# Ayarlar
st.set_page_config(page_title="Emlak Danışmanı v1.0", layout="wide")

# =============================================================================
# SEGMENT MODELLERİ YAPILANDIRMASI
# =============================================================================
SEGMENT_CONFIG = {
    '0-500K': {
        'model_file': 'segment_model_0_500K_20251214_0956.joblib',
        'min_price': 0,
        'max_price': 500_000
    },
    '500K-1M': {
        'model_file': 'segment_model_500K_1M_20251214_0956.joblib',
        'min_price': 500_000,
        'max_price': 1_000_000
    },
    '1M-2M': {
        'model_file': 'segment_model_1M_2M_20251214_0956.joblib',
        'min_price': 1_000_000,
        'max_price': 2_000_000
    },
    '2M-5M': {
        'model_file': 'segment_model_2M_5M_20251214_0956.joblib',
        'min_price': 2_000_000,
        'max_price': 5_000_000
    }
}

@st.cache_resource
def load_model():
    """Genel XGBoost modelini yükle (ön tahmin için)"""
    try:
        model = joblib.load('xgboost_model_r2_82pct_20251214_0952.joblib')
        return model
    except Exception as e:
        st.error(f"Genel model yüklenemedi: {e}")
        return None

@st.cache_resource
def load_segment_models():
    """Tüm segment modellerini yükle"""
    segment_models = {}
    for segment_name, config in SEGMENT_CONFIG.items():
        try:
            segment_models[segment_name] = joblib.load(config['model_file'])
        except Exception as e:
            st.warning(f"Segment modeli yüklenemedi ({segment_name}): {e}")
            segment_models[segment_name] = None
    return segment_models

def get_segment_for_price(price):
    """Fiyata göre uygun segmenti belirle"""
    for segment_name, config in SEGMENT_CONFIG.items():
        if config['min_price'] <= price < config['max_price']:
            return segment_name
    # 5M üzeri için en yüksek segmenti kullan
    if price >= 5_000_000:
        return '2M-5M'
    return '0-500K'

@st.cache_data
def load_model_metadata():
    """Model metadata dosyasını yükle"""
    try:
        with open('xgboost_model_r2_82pct_20251214_0952_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"Model metadata yüklenemedi: {e}")
        return None

@st.cache_data
def load_processed_data():
    """İşlenmiş veri setini yükle"""
    try:
        df = pd.read_csv('processed_dataset_with_features.csv', low_memory=False)
        return df
    except Exception as e:
        st.error(f"İşlenmiş veri seti yüklenemedi: {e}")
        return None

# Modelleri yükle
model = load_model()
segment_models = load_segment_models()
model_metadata = load_model_metadata()
df_processed = load_processed_data()

# Dil Seçeneği
if 'language' not in st.session_state:
    st.session_state.language = 'TR'

with st.sidebar:
    st.write("Ayarlar / Settings")
    
    if st.button("Dil Değiştir / Switch Language"):
        st.session_state.language = 'EN' if st.session_state.language == 'TR' else 'TR'
    st.caption(f"Aktif Dil: {st.session_state.language}")

# Metin Sözlüğü (Basit lokalleştirme)
TEXTS = {
    'TR': {
        'title': "Emlak Yatırım Danışmanı",
        'subtitle': "Destekli Fiyat Analiz Sistemi",
        'sidebar_header': "Ev Özellikleri",
        'district': "İlçe",
        'neighborhood': "Mahalle",
        'net_m2': "Net m²",
        'rooms': "Oda Sayısı",
        'age': "Bina Yaşı",
        'floor': "Kat Durumu",
        'heating': "Isıtma Tipi",
        'listing_price': "İlan Fiyatı (TL)",
        'analyze_btn': "ANALİZ ET (Fırsat mı?)",
        'loading': "Yapay Zeka Modeli Hesaplanıyor...",
        'result_fair': "Hesaplanan Adil Değer",
        'result_listing': "İlandaki Fiyat",
        'status_opp': "FIRSAT",
        'status_exp': "PAHALI",
        'status_norm': "NORMAL"
    },
    'EN': {
        'title': "Real Estate Advisor",
        'subtitle': "Price Analysis System",
        'sidebar_header': "Property Features",
        'district': "District",
        'neighborhood': "Neighborhood",
        'net_m2': "Net m²",
        'rooms': "Rooms",
        'age': "Building Age",
        'floor': "Floor",
        'heating': "Heating",
        'listing_price': "Listing Price (TL)",
        'analyze_btn': "ANALYZE (Is it a deal?)",
        'loading': "AI Model is Calculating...",
        'result_fair': "Calculated Fair Value",
        'result_listing': "Listing Price",
        'status_opp': "OPPORTUNITY",
        'status_exp': "EXPENSIVE",
        'status_norm': "NORMAL"
    }
}

t = TEXTS[st.session_state.language]

@st.cache_data
def get_dropdown_options(filepath):
    """Dropdown seçeneklerini yükle"""
    try:
        cols = ['District', 'Neighborhood', 'Number of rooms', 'Building Age', 'Floor location', 'Heating', 'Available for Loan']
        df = pd.read_csv(filepath, sep=';', usecols=cols)
        df = df[df['Available for Loan'] == 'Yes']
        
        return df
    except Exception as e:
        st.error(f"Veri dosyası (hackathon_train_set.csv) bulunamadı! Hata: {e}")
        return pd.DataFrame()

df = get_dropdown_options('hackathon_train_set.csv')

def sort_numeric_strings(values):
    """Sayısal sıralama"""
    def extract_number(val):
        val_str = str(val)
        numbers = re.findall(r'\d+', val_str)
        if numbers:
            return int(numbers[0])
        if 'New' in val_str or 'Sıfır' in val_str or '0' == val_str:
            return 0
        if 'Ground' in val_str or 'Giriş' in val_str or 'Zemin' in val_str:
            return 0
        if 'Basement' in val_str or 'Bodrum' in val_str:
            return -1
        return 999
    return sorted(values, key=extract_number)

def sort_floor_options(values):
    """Kat sıralaması"""
    floor_order = {
        'Basement': -2, 'Kot 1': -1, 'Kot 2': -1, 'Kot 3': -1, 'Kot 4': -1,
        'Ground floor': 0, 'Garden Floor': 0, 'Entrance floor': 0, 'High entrance': 0,
    }
    def get_floor_order(val):
        val_str = str(val)
        if val_str in floor_order:
            return (floor_order[val_str], val_str)
        numbers = re.findall(r'\d+', val_str)
        if numbers:
            num = int(numbers[0])
            if '30' in val_str and 'more' in val_str:
                return (30, val_str)
            return (num, val_str)
        if 'Penthouse' in val_str: return (100, val_str)
        if 'Villa' in val_str: return (101, val_str)
        if 'Private' in val_str: return (102, val_str)
        return (999, val_str)
    return sorted(values, key=get_floor_order)

# Sidebar
st.sidebar.header(t['sidebar_header'])

if not df.empty:
    districts = sorted(df['District'].unique())
    selected_district = st.sidebar.selectbox(t['district'], districts)
    
    neighborhoods = sorted(df[df['District'] == selected_district]['Neighborhood'].unique())
    selected_neighborhood = st.sidebar.selectbox(t['neighborhood'], neighborhoods)
    
    rooms = st.sidebar.selectbox(t['rooms'], sort_numeric_strings(df['Number of rooms'].unique()))
    net_m2 = st.sidebar.number_input(t['net_m2'], min_value=10, max_value=500, value=90)
    age = st.sidebar.selectbox(t['age'], sort_numeric_strings(df['Building Age'].unique()))
    floor = st.sidebar.selectbox(t['floor'], sort_floor_options(df['Floor location'].astype(str).unique()))
    heating = st.sidebar.selectbox(t['heating'], sorted(df['Heating'].unique()))
    
    price_input = st.sidebar.number_input(t['listing_price'], min_value=50000, value=500000, step=10000)
else:
    st.error("Veri seti bulunamadı!")
    st.stop()

# Ana ekran
st.title(t['title'])
st.markdown(f"_{t['subtitle']}_")

with st.expander("Seçilen Özellikler / Selected Features"):
    st.write({
        "İlçe": selected_district if not df.empty else "-",
        "Mahalle": selected_neighborhood if not df.empty else "-",
        "m2": net_m2,
        "Isıtma": heating if not df.empty else "-"
    })

if st.button(t['analyze_btn'], type="primary"):
    with st.spinner(t['loading']):
        time.sleep(0.5)
        
        if model is None or df_processed is None or model_metadata is None:
            st.error("Model, metadata veya veri seti yüklenemedi!")
        else:
            try:
                feature_columns = model_metadata['feature_columns']
                dist_col = f'Dist_{selected_district}'
                dist_cols = [col for col in feature_columns if col.startswith('Dist_')]
                template_row = df_processed.iloc[0:1].copy()
                
                for dcol in dist_cols:
                    if dcol in template_row.columns:
                        template_row[dcol] = False
                if dist_col in template_row.columns:
                    template_row[dist_col] = True
                
                df_train = pd.read_csv('hackathon_train_set.csv', sep=';')
                df_train['Price_Num'] = df_train['Price'].astype(str).str.replace(' TL', '').str.replace('.', '').astype(float)
                neigh_mean = df_train[df_train['Neighborhood'] == selected_neighborhood]['Price_Num'].mean()
                if pd.isna(neigh_mean):
                    neigh_mean = df_train['Price_Num'].mean()
                template_row['Neighborhood_TargetEncoded'] = neigh_mean
                template_row['Neighborhood_TE_Normalized'] = neigh_mean / df_train['Price_Num'].max()
                template_row['Neighborhood_TE_Log'] = np.log1p(neigh_mean)
                
                gross_m2 = net_m2 * 1.1
                template_row['m² (Gross)'] = gross_m2
                template_row['m² (Net)'] = net_m2
                
                def parse_rooms(x):
                    try:
                        parts = str(x).split('+')
                        return sum([float(p) for p in parts])
                    except:
                        return 3.0
                total_rooms = parse_rooms(rooms)
                template_row['Total_Rooms'] = total_rooms
                
                age_map = {
                    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                    '5-10 between': 7.5, '11-15 between': 13,
                    '16-20 between': 18, '21-25 between': 23,
                    '26-30 between': 28, '31  and more than': 35
                }
                building_age = age_map.get(age, 10)
                template_row['Building_Age_Numeric'] = building_age
                
                def parse_floor(x):
                    x = str(x).lower()
                    if 'garden' in x or 'ground' in x or 'entrance' in x: return 0
                    elif 'basement' in x: return -1
                    try:
                        return int(re.findall(r'-?\d+', x)[0])
                    except:
                        return 2
                floor_num = parse_floor(floor)
                template_row['Floor_Numeric'] = floor_num
                num_floors = max(5, floor_num + 2)
                template_row['Number of floors'] = num_floors
                template_row['Heating'] = heating
                
                template_row['Efficiency_Ratio'] = net_m2 / gross_m2
                template_row['Sqm_per_room'] = net_m2 / total_rooms if total_rooms > 0 else net_m2
                template_row['Age_Floor_interaction'] = building_age * floor_num
                template_row['Interaction_Score'] = gross_m2 * neigh_mean
                template_row['Spaciousness'] = net_m2 / total_rooms if total_rooms > 0 else net_m2
                
                template_row['m² (Gross)_squared'] = gross_m2 ** 2
                template_row['m² (Gross)_sqrt'] = np.sqrt(gross_m2)
                template_row['m² (Gross)_log'] = np.log1p(gross_m2)
                template_row['m² (Net)_squared'] = net_m2 ** 2
                template_row['m² (Net)_sqrt'] = np.sqrt(net_m2)
                template_row['m² (Net)_log'] = np.log1p(net_m2)
                template_row['Total_Rooms_squared'] = total_rooms ** 2
                template_row['Total_Rooms_sqrt'] = np.sqrt(total_rooms)
                template_row['Total_Rooms_log'] = np.log1p(total_rooms)
                template_row['Building_Age_Numeric_squared'] = building_age ** 2
                template_row['Building_Age_Numeric_sqrt'] = np.sqrt(building_age)
                template_row['Building_Age_Numeric_log'] = np.log1p(building_age)
                template_row['Floor_Numeric_squared'] = floor_num ** 2
                template_row['Floor_Numeric_sqrt'] = np.sqrt(abs(floor_num))
                template_row['Floor_Numeric_log'] = np.log1p(abs(floor_num))
                template_row['Floor_Ratio'] = floor_num / num_floors if num_floors > 0 else 0
                template_row['Available_for_Loan_Bin'] = 1
                
                input_df = template_row[feature_columns].copy()
                for col in input_df.columns:
                    if input_df[col].dtype == 'object':
                        input_df[col] = input_df[col].astype('category')
                
                # 1. Adım: Genel model ile ön tahmin yap
                log_prediction_initial = model.predict(input_df)[0]
                initial_price = np.expm1(log_prediction_initial)
                
                # 2. Adım: Ön tahmine göre uygun segmenti belirle
                segment = get_segment_for_price(initial_price)
                segment_model = segment_models.get(segment)
                
                # 3. Adım: Segment modeli varsa, onunla final tahmin yap
                if segment_model is not None:
                    try:
                        log_prediction = segment_model.predict(input_df)[0]
                        predicted_price = np.expm1(log_prediction)
                        used_segment = segment
                    except Exception as seg_error:
                        # Segment modeli hata verirse genel modeli kullan
                        predicted_price = initial_price
                        used_segment = "Genel Model"
                else:
                    # Segment modeli yoksa genel modeli kullan
                    predicted_price = initial_price
                    used_segment = "Genel Model"
                
            except Exception as e:
                st.error(f"Tahmin hatası: {e}")
                predicted_price = net_m2 * 5000
                if "New" in str(age) or "0" in str(age): predicted_price *= 1.2
                if "Natural Gas" in str(heating): predicted_price *= 1.1
                used_segment = "Fallback"

            col1, col2, col3 = st.columns(3)
            
            diff = price_input - predicted_price
            ratio = diff / predicted_price
            
            with col1:
                st.metric(t['result_fair'], f"{predicted_price:,.0f} TL")
                st.caption(f"📊 Segment: {used_segment}")
            
            with col2:
                st.metric(t['result_listing'], f"{price_input:,.0f} TL", delta=f"{ratio:.1%}")
            
            with col3:
                if price_input < predicted_price * 0.9:
                    st.success(f"### {t['status_opp']}")
                elif price_input > predicted_price * 1.1:
                    st.error(f"### {t['status_exp']}")
                else:
                    st.warning(f"### {t['status_norm']}")
