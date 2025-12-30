import streamlit as st
from PIL import Image
from fpdf import FPDF
from DRD_app_backend import analyze_image_backend

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("css/style.css")

st.set_page_config(
    page_title="CNN-SVM Hybrid model",
    page_icon="icon.png",
)

st.markdown("""
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 0rem !important;}
        .main {margin-top: 0px !important;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<body>', unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="title">Diabetic Retinopathy Detector</div>', unsafe_allow_html=True)

# ---------- Image Upload ----------
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="section-label" style="margin-top:40px">Image</div>', unsafe_allow_html=True)
with col2:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "tif"])

st.markdown('<br>', unsafe_allow_html=True)

# ---------- Image Preview ----------
preview_col = st.columns([1, 6, 1])[1]

with preview_col:
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.markdown('<div class="preview-placeholder">Upload an image to preview</div>', unsafe_allow_html=True)

# ---------- Button Design ----------
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

# ---------- Color Map based on Severity ----------
SEVERITY_COLORS = {
    "No DR": "#045827",          
    "Mild": "#6f5902",           
    "Moderate": "#8a4407",       
    "Severe": "#d02815",         
    "Proliferative": "#5a0202"   
}

# ---------- PDF Generator ----------
def generate_pdf(prediction, confidence, model):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Diabetic Retinopathy Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)

    output = "DR_Report.pdf"
    pdf.output(output)
    return output

# ---------- Results Section ----------
label_col, result_col = st.columns([1, 3])

with label_col:
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

with result_col:

    if analyze_clicked and uploaded_file:

        with st.spinner("Analyzing image using models... Please wait..."):
            result_KDR, result_MS = analyze_image_backend(image)

        tab1, tab2 = st.tabs(["Model Trained on KDR Dataset", "Model Trained on MS Dataset"])

        # ----- Function to display each result -----
        def show_result(result_dict):
            severity = result_dict["disease"]
            conf = result_dict["confidence"]
            model_name = result_dict["model"]
            color = SEVERITY_COLORS.get(severity, "#ffffff")

            st.markdown(f"""
                <div style="
                    background-color:#f4e8ff;
                    padding:20px;
                    border-radius:10px;
                    border-left:10px solid {color};
                    font-size:16px;
                    color: #410B49;">
                    <b>Disease Level : </b> 
                    <span style="color:{color}; font-weight:bold">{severity}</span><br>
                    <b>Prediction Confidence : </b> {conf:.2%}<br>
                </div>
                <br>
            """, unsafe_allow_html=True)

            # PDF Download
            pdf_file = generate_pdf(severity, conf, model_name)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name=f"DR_Report_{model_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key=f"download_{model_name.replace(' ', '_')}" 
                )


        # ----- Show results inside tabs -----
        with tab1:
            show_result(result_KDR)

        with tab2:
            show_result(result_MS)

    elif analyze_clicked and not uploaded_file:
        st.error("Please upload an image first!")

st.markdown('</body>', unsafe_allow_html=True)
