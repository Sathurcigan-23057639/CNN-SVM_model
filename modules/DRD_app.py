import streamlit as st
from PIL import Image

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("css/style.css")

st.markdown("""
    <style>
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem !important;
        }
        .main {
            margin-top: 0px !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


st.markdown('<body>', unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Diabetic Retinopathy Detector</div>', unsafe_allow_html=True)

# Upload Section
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="section-label" style="margin-top:40px">Image</div>', unsafe_allow_html=True)
with col2:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "tif"])

st.markdown('<br>', unsafe_allow_html=True)

# Preview Box
preview_col = st.columns([1, 6, 1])[1]

with preview_col:
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.markdown('<div class="preview-placeholder">Upload an image to preview</div>', unsafe_allow_html=True)


# Result Section
btn_left, btn_center, btn_right = st.columns([2, 1, 2])
with btn_center:
    st.markdown("""
                    <style>
                        div.stButton > button:first-child {
                            color: rgb(65, 11, 73);
                            background-color: rgb(215, 212, 224);
                            width: 100px;
                            font-weight: bold;
                            margin-top: 20px;
                        }
                    </style>
                """, unsafe_allow_html=True)
    analyze_clicked = st.button("Analyze")

st.markdown('<br>', unsafe_allow_html=True)

# Result row: label and result box
label_col, result_col = st.columns([1, 3])

with label_col:
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

with result_col:
    if analyze_clicked and uploaded_file:
        prediction = "Moderate Diabetic Retinopathy"
        confidence = 0.87
        st.markdown(f"**Prediction:** {prediction}<br>**Confidence:** {confidence:.2%}", unsafe_allow_html=True)
    # else:
    #     st.markdown('<div style="margin-top:15px">Result will appear here</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</body>', unsafe_allow_html=True)
