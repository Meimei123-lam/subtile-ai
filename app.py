import os
import cv2
import requests
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from simple_lama_inpainting import SimpleLama

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(
    page_title="AI Video Subtitle Inpainter & Translator",
    page_icon="🎬",
    layout="wide"
)

# --- TẢI HOẶC CHUẨN BỊ FONT TIẾNG VIỆT ---
def get_vietnamese_font(size=26):
    font_path = "Roboto-Bold.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf"
        r = requests.get(url, timeout=10)
        with open(font_path, "wb") as f:
            f.write(r.content)
    try:
        return ImageFont.truetype(font_path, size=size)
    except Exception:
        return ImageFont.load_default()

# --- KHỞI TẠO CÁC MODEL AI ---
@st.cache_resource
def load_all_models():
    # Model OCR quét text và lấy tọa độ hộp chữ
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    # Model AI Inpainting tự vẽ xóa chữ (Generative Fill)
    try:
        lama = SimpleLama()
    except Exception:
        lama = None
    # Model Whisper nhận diện giọng nói
    whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return ocr, lama, whisper

ocr_model, lama_model, whisper_model = load_all_models()
translator = GoogleTranslator(source='auto', target='vi')

# --- HÀM PHÂN TÍCH VIDEO ---
def analyze_video(video_path, sample_rate_sec=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(fps * sample_rate_sec))
    
    sub_data = []
    current_frame = 0
    pbar = st.progress(0, text="AI đang quét vị trí chữ và dịch tự động...")
    
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = current_frame / fps
        result = ocr_model.ocr(frame, cls=False)
        
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0].strip()
                conf = line[1][1]
                
                if conf > 0.6 and len(text) > 1:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    xmin, xmax = int(min(xs)), int(max(xs))
                    ymin, ymax = int(min(ys)), int(max(ys))
                    
                    try:
                        trans_text = translator.translate(text)
                    except Exception:
                        trans_text = text
                    
                    sub_data.append({
                        "Start": round(timestamp, 2),
                        "End": round(timestamp + sample_rate_sec, 2),
                        "Original": text,
                        "Translation": trans_text,
                        "X": xmin,
                        "Y": ymin,
                        "W": xmax - xmin,
                        "H": ymax - ymin
                    })
        
        current_frame += step_frames
        if current_frame >= total_frames:
            break
        pbar.progress(min(current_frame / total_frames, 1.0))
        
    cap.release()
    pbar.empty()
    return pd.DataFrame(sub_data)

# --- HÀM XÓA TEXT VÀ VẼ LẠI SUB MỚI ---
def process_frame(frame, sub_rows, use_generative_fill=True):
    h, w, _ = frame.shape
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 1. BƯỚC XÓA CHỮ (INPAINTING)
    if use_generative_fill and lama_model is not None:
        # Tạo mask đen trắng cho các vùng chữ
        mask = np.zeros((h, w), dtype=np.uint8)
        for _, row in sub_rows.iterrows():
            x, y, bw, bh = int(row['X']), int(row['Y']), int(row['W']), int(row['H'])
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            mask[y0:y1, x0:x1] = 255
            
        mask_pil = Image.fromarray(mask)
        # AI tự sinh lại hình ảnh nền
        img_pil = lama_model(img_pil, mask_pil)
    else:
        # Phương án dự phòng nếu không dùng GPU: vẽ nền mờ che chữ
        draw_temp = ImageDraw.Draw(img_pil)
        for _, row in sub_rows.iterrows():
            x, y, bw, bh = int(row['X']), int(row['Y']), int(row['W']), int(row['H'])
            draw_temp.rectangle([x-2, y-2, x+bw+2, y+bh+2], fill=(15, 15, 15))
            
    # 2. BƯỚC VẼ SUB TIẾNG VIỆT MỚI
    draw = ImageDraw.Draw(img_pil)
    font = get_vietnamese_font(size=24)
    
    for _, row in sub_rows.iterrows():
        x, y = int(row['X']), int(row['Y'])
        text_vn = str(row['Translation'])
        
        # Tạo viền đen (outline) cho chữ để dễ nhìn
        outline_color = (0, 0, 0)
        for dx, dy in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (1,1)]:
            draw.text((x + dx, y + dy), text_vn, font=font, fill=outline_color)
            
        # Vẽ chữ chính màu vàng/trắng phong cách phụ đề điện ảnh
        draw.text((x, y), text_vn, font=font, fill=(255, 230, 100))
        
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- GIAO DIỆN CHÍNH (STREAMLIT WEB) ---
st.title("🎬 AI Video Subtitle Cleaner, Translator & Replacer")
st.markdown("Xóa sạch sub cũ tự động bằng **Generative Inpainting** và chèn đè bản dịch chuẩn vị trí.")

# Thanh bên trái
st.sidebar.header("⚙️ Cài đặt & Tải video")
uploaded_file = st.sidebar.file_uploader("Chọn file video:", type=["mp4", "mov", "avi"])
use_lama = st.sidebar.checkbox("Bật AI Generative Fill (LaMa)", value=True, help="Tự động vẽ lại cảnh nền sau khi xóa text gốc")
sample_rate = st.sidebar.slider("Tần suất quét (giây/lần):", 0.5, 3.0, 1.0, 0.5)

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.sidebar.video(video_path)

    if "sub_df" not in st.session_state:
        if st.sidebar.button("🚀 Bắt đầu quét & dịch phụ đề"):
            df = analyze_video(video_path, sample_rate_sec=sample_rate)
            st.session_state["sub_df"] = df
            st.rerun()

    if "sub_df" in st.session_state:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Bảng chỉnh sửa bản dịch thủ công")
            st.info("Nhấp đúp vào cột **Translation** để sửa câu chữ. Video bên phải sẽ lập tức cập nhật!")
            
            edited_df = st.data_editor(
                st.session_state["sub_df"],
                num_rows="dynamic",
                use_container_width=True,
                height=520
            )
            st.session_state["sub_df"] = edited_df

        with col2:
            st.subheader("👁️ Xem trước khung hình (Preview)")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            
            preview_sec = st.slider("Di chuyển timeline để kiểm tra (giây):", 0.0, float(total_sec), 1.0, 0.5)
            
            cap.set(cv2.CAP_PROP_POS_MSEC, preview_sec * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                active_subs = edited_df[(edited_df["Start"] <= preview_sec) & (edited_df["End"] >= preview_sec)]
                if not active_subs.empty:
                    preview_frame = process_frame(frame, active_subs, use_generative_fill=use_lama)
                else:
                    preview_frame = frame
                
                st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            if st.button("💾 Xuất toàn bộ video hoàn chỉnh"):
                out_path = os.path.join(tempfile.gettempdir(), "final_subbed_video.mp4")
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                
                p_render = st.progress(0, text="Đang render video thành phẩm...")
                frame_idx = 0
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    t_cur = frame_idx / fps
                    subs = edited_df[(edited_df["Start"] <= t_cur) & (edited_df["End"] >= t_cur)]
                    if not subs.empty:
                        frame = process_frame(frame, subs, use_generative_fill=use_lama)
                    out.write(frame)
                    frame_idx += 1
                    if frame_idx % 25 == 0:
                        p_render.progress(min(frame_idx / total_f, 1.0))
                        
                cap.release()
                out.release()
                p_render.empty()
                st.success("🎉 Xuất video hoàn tất!")
                st.video(out_path)