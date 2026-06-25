import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import time
import os
import psutil

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Age & Gender Classification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CONSTANTS ----
GENDER_CLASSES = ['Nữ (Female)', 'Nam (Male)']
AGE_CLASSES = ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
DEFAULT_IMAGE_SIZE = 227

# ---- STYLING ----
# Custom CSS for Premium Glassmorphism Look & Typography
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%);
        color: #f8fafc;
    }

    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .glass-header {
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
        opacity: 0.95 !important;
    }

    /* Custom sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---- STATE INITIALIZATION ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- MODEL LOADING UTILITY ----
@st.cache_resource
def get_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Find available models in outputs directory
models_dir = "outputs"
available_models = []
if os.path.exists(models_dir):
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.h5')]

# Default if empty
if not available_models:
    available_models = ["No models found in outputs/"]

# ---- SIDEBAR DESIGN ----
st.sidebar.markdown("<h2 class='glass-header'>🛠️ Cấu hình mô hình</h2>", unsafe_allow_html=True)

# Select Model
selected_model_name = st.sidebar.selectbox(
    "Chọn file weights (.h5):",
    available_models,
    index=0
)

# Load selected model
model = None
model_path = os.path.join(models_dir, selected_model_name) if selected_model_name != "No models found in outputs/" else ""

if model_path and os.path.exists(model_path):
    model = get_model(model_path)
    if isinstance(model, str):
        st.sidebar.error(model)
        model = None
    else:
        st.sidebar.success(f"✓ Đã tải mô hình: {selected_model_name}")
else:
    st.sidebar.warning("⚠️ Vui lòng huấn luyện mô hình hoặc đặt file weights (.h5) vào thư mục `outputs/`.")

# Sidebar Metrics
st.sidebar.markdown("<br><hr style='border: 0.5px solid rgba(255,255,255,0.1)'><br>", unsafe_allow_html=True)
st.sidebar.markdown("### 📊 Trạng thái hệ thống")

cpu_use = psutil.cpu_percent()
mem_use = psutil.virtual_memory().percent

st.sidebar.progress(cpu_use / 100.0, text=f"CPU Usage: {cpu_use}%")
st.sidebar.progress(mem_use / 100.0, text=f"RAM Usage: {mem_use}%")

# Main Page Layout
st.markdown("<h1 class='glass-header' style='font-size: 3rem; text-align: center;'>🎯 Nhận diện Tuổi & Giới tính</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Hệ thống phân loại đa nhiệm sử dụng mạng Neural tích chập Deep Learning (Keras/TensorFlow)</p>", unsafe_allow_html=True)

# Tabs
tab_predict, tab_eda, tab_history = st.tabs(["🔮 Dự đoán (Prediction)", "📊 Phân tích dữ liệu (EDA)", "📝 Lịch sử (History)"])

# ---- PREPROCESSING PIPELINE ----
def preprocess_image(pil_img, target_size=DEFAULT_IMAGE_SIZE):
    # Resize to 256x256
    img = pil_img.convert('RGB').resize((256, 256))
    
    # Center Crop to target_size x target_size
    left = (256 - target_size) // 2
    top = (256 - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    img_cropped = img.crop((left, top, right, bottom))
    
    # Normalize
    img_array = np.array(img_cropped) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)
    
    return img_expanded, img_cropped

# ---- TAB 1: PREDICTION ----
with tab_predict:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    # Input options columns
    col_input, col_display = st.columns([2, 3])
    
    with col_input:
        st.markdown("### 📥 Chọn nguồn ảnh")
        input_type = st.radio("Nguồn ảnh đầu vào:", ["Tải ảnh lên (Upload File)", "Chụp từ Webcam (Webcam)", "Ảnh mẫu (Sample Images)"])
        
        uploaded_file = None
        img_input = None
        
        if input_type == "Tải ảnh lên (Upload File)":
            uploaded_file = st.file_uploader("Kéo thả hoặc chọn tệp ảnh của bạn (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                img_input = Image.open(uploaded_file)
                
        elif input_type == "Chụp từ Webcam (Webcam)":
            webcam_image = st.camera_input("Chụp ảnh từ webcam của bạn")
            if webcam_image:
                img_input = Image.open(webcam_image)
                
        elif input_type == "Ảnh mẫu (Sample Images)":
            sample_dir = "samples"
            os.makedirs(sample_dir, exist_ok=True)
            samples = {
                "Nữ trẻ tuổi (Female Sample)": os.path.join(sample_dir, "sample_female.png"),
                "Nam thanh niên (Male Sample)": os.path.join(sample_dir, "sample_male.png"),
                "Trẻ em (Child Sample)": os.path.join(sample_dir, "sample_child.png"),
                "Người lớn tuổi (Elderly Sample)": os.path.join(sample_dir, "sample_elderly.png")
            }
            
            # Check if samples exist, if not show options
            avail_samples = {k: v for k, v in samples.items() if os.path.exists(v)}
            
            if avail_samples:
                selected_sample = st.selectbox("Chọn ảnh mẫu để kiểm thử:", list(avail_samples.keys()))
                img_input = Image.open(avail_samples[selected_sample])
            else:
                st.warning("⚠️ Không tìm thấy ảnh mẫu trong thư mục `samples/`. Bạn có thể tải ảnh lên hoặc dùng webcam.")
                # Fallback to general samples
                
        # Predict Button
        predict_button = st.button("🚀 Thực hiện Nhận diện")
        
    with col_display:
        if img_input:
            st.markdown("### 🖼️ Ảnh đang chọn")
            st.image(img_input, use_container_width=True, caption="Ảnh gốc")
        else:
            st.info("ℹ️ Vui lòng chọn nguồn ảnh ở cột bên trái để tiếp tục.")
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results Area
    if predict_button and img_input:
        if model is None:
            st.error("❌ Không thể dự đoán do mô hình chưa được tải thành công. Vui lòng kiểm tra lại cấu hình bên thanh Sidebar.")
        else:
            with st.spinner("Đang tiền xử lý ảnh và chạy suy diễn mạng Neural..."):
                # Preprocess image
                t0 = time.time()
                input_tensor, cropped_img = preprocess_image(img_input)
                
                # Predict
                preds = model.predict(input_tensor)
                latency = (time.time() - t0) * 1000 # ms
                
                # Keras model outputs could be in any order, check names
                output_names = [out.name for out in model.outputs]
                # Default logic: first output gender, second output age (based on model.py)
                gender_pred = preds[0]
                age_pred = preds[1]
                
                # Match predictions to actual names if outputs list is swapped
                for i, name in enumerate(output_names):
                    if 'gender' in name:
                        gender_pred = preds[i]
                    elif 'age' in name:
                        age_pred = preds[i]
                
                # Best index
                gender_idx = np.argmax(gender_pred)
                age_idx = np.argmax(age_pred)
                
                gender_conf = gender_pred[0][gender_idx]
                age_conf = age_pred[0][age_idx]
                
                # Display Results in Glass Cards
                st.markdown("<h3 class='glass-header'>✨ Kết quả Nhận diện</h3>", unsafe_allow_html=True)
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f"""
                    <div class='glass-card' style='text-align: center;'>
                        <div class='metric-label'>Giới tính (Gender)</div>
                        <div class='metric-value'>{GENDER_CLASSES[gender_idx]}</div>
                        <div style='color: #a855f7; font-weight: 600;'>Độ tin cậy: {gender_conf:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    st.markdown(f"""
                    <div class='glass-card' style='text-align: center;'>
                        <div class='metric-label'>Nhóm tuổi (Age Group)</div>
                        <div class='metric-value'>{AGE_CLASSES[age_idx]} tuổi</div>
                        <div style='color: #a855f7; font-weight: 600;'>Độ tin cậy: {age_conf:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_res3:
                    st.markdown(f"""
                    <div class='glass-card' style='text-align: center;'>
                        <div class='metric-label'>Thời gian xử lý</div>
                        <div class='metric-value'>{latency:.1f} ms</div>
                        <div style='color: #10b981; font-weight: 600;'>Tốc độ cực nhanh</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add to history
                history_entry = {
                    "Thời gian": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Giới tính": GENDER_CLASSES[gender_idx].split(' ')[0],
                    "Độ tin cậy Giới tính": f"{gender_conf:.2%}",
                    "Nhóm tuổi": AGE_CLASSES[age_idx],
                    "Độ tin cậy Tuổi": f"{age_conf:.2%}",
                    "Độ trễ (ms)": f"{latency:.1f}ms"
                }
                st.session_state.history.append(history_entry)
                
                # Plotly Probability Charts
                st.markdown("### 📊 Biểu đồ Phân bố Xác suất (Probability Distributions)")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_gender = go.Figure(go.Bar(
                        x=gender_pred[0],
                        y=GENDER_CLASSES,
                        orientation='h',
                        marker=dict(color=['#ec4899', '#3b82f6']),
                        text=[f"{p:.2%}" for p in gender_pred[0]],
                        textposition='auto'
                    ))
                    fig_gender.update_layout(
                        title="Phân bố xác suất Giới tính (Gender Probability)",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(range=[0, 1]),
                        height=250
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)
                    
                with col_chart2:
                    fig_age = go.Figure(go.Bar(
                        x=age_pred[0],
                        y=AGE_CLASSES,
                        orientation='h',
                        marker=dict(color=age_pred[0], colorscale='Viridis'),
                        text=[f"{p:.2%}" for p in age_pred[0]],
                        textposition='inside'
                    ))
                    fig_age.update_layout(
                        title="Phân bố xác suất Nhóm tuổi (Age Group Probability)",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(range=[0, 1]),
                        height=350
                    )
                    st.plotly_chart(fig_age, use_container_width=True)

# ---- TAB 2: DATA ANALYSIS (EDA) ----
with tab_eda:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glass-header'>📊 Tổng quan Bộ dữ liệu Adience</h3>", unsafe_allow_html=True)
    st.markdown("""
    Bộ dữ liệu **Adience** là bộ dữ liệu thực tế thu thập từ Flickr để phân loại nhóm tuổi và giới tính. 
    Các bức ảnh được chụp trong môi trường thực (in-the-wild), không có sắp đặt ánh sáng hoặc góc quay cố định, khiến bài toán trở nên thử thách nhưng có tính thực tiễn cực kỳ cao.
    
    #### Thống kê tổng quan bộ dữ liệu:
    - **Tổng số ảnh khuôn mặt**: ~26,580 ảnh của 2,284 người dùng khác nhau.
    - **Được aligned**: Đã được căn chỉnh và lưu trong định dạng aligned (227x227 pixels).
    - **Chia folds**: Được chia làm 5 folds (từ fold 0 đến fold 4) để đánh giá chéo (cross-validation).
    """)
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        # Age breakdown representation
        age_counts = [2487, 1823, 2197, 2824, 5132, 2821, 1422, 1374]
        fig_eda_age = go.Figure(go.Pie(
            labels=AGE_CLASSES,
            values=age_counts,
            hole=0.4,
            marker=dict(colorscale='Portland')
        ))
        fig_eda_age.update_layout(
            title="Tỷ lệ phân bố các Nhóm tuổi trong Adience Dataset",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350
        )
        st.plotly_chart(fig_eda_age, use_container_width=True)
        
    with col_stat2:
        # Gender breakdown representation
        gender_counts = [10452, 9864]
        fig_eda_gender = go.Figure(go.Bar(
            x=['Nữ (f)', 'Nam (m)'],
            y=gender_counts,
            marker=dict(color=['#ec4899', '#3b82f6']),
            text=[f"{count:,}" for count in gender_counts],
            textposition='auto'
        ))
        fig_eda_gender.update_layout(
            title="Phân bố Giới tính trong Adience Dataset",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350
        )
        st.plotly_chart(fig_eda_gender, use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ---- TAB 3: HISTORY ----
with tab_history:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glass-header'>📝 Lịch sử nhận diện trong phiên làm việc</h3>", unsafe_allow_html=True)
    
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        # Download button
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải xuống báo cáo lịch sử (CSV)",
            data=csv_data,
            file_name="predict_history.csv",
            mime="text/csv"
        )
        
        # Clear button
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("ℹ️ Chưa có lịch sử nhận diện nào được lưu. Hãy thử dự đoán một vài hình ảnh!")
        
    st.markdown("</div>", unsafe_allow_html=True)
