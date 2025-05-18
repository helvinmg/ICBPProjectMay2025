from PIL import Image
import streamlit as st
import os
import time
import importlib
import nst
importlib.reload(nst)
from nst import load_image, run_style_transfer, im_convert
from realesrgan_wrapper import upscale_image

BASE_DIR = "."
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
STYLED_DIR = os.path.join(BASE_DIR, "outputs", "styled")
UPSCALED_DIR = os.path.join(BASE_DIR, "outputs", "upscaled")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(STYLED_DIR, exist_ok=True)
os.makedirs(UPSCALED_DIR, exist_ok=True)

st.set_page_config(page_title="ICBP Project", layout="wide")

# Combine all CSS into one block with stronger styling
st.markdown(
    """
    <style>
    /* Expand the main container to use more screen width */
    .main .block-container {
        max-width: 1400px !important;
        padding-left: 1rem;
        padding-right: 1rem;
        width: 100% !important;
        padding-top: 0 !important; /* Remove ALL top padding */
    }
    /* Ensure content spans full width */
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        width: 100% !important;
    }
    /* Control image size */
    .stImage img {
        max-height: 350px !important; 
        width: auto !important;
    }
    /* Override Streamlit's default footer styling */
    footer {
        visibility: visible !important;
        display: block !important;
        position: relative !important;
        margin-top: 70px !important;
        clear: both !important;
    }
    footer:after {
        content: '© HMG';
        display: block;
        position: relative;
        color: grey;
        padding: 10px 0;
        top: 5px;
    }
    /* Aggressive title spacing fixes */
    h1:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
        text-align: center !important;
    }
    /* Target the specific wrapper div that contains the title */
    .css-1dp5vir {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* Target all wrappers that might contain the title */
    div.css-1544g2n.e1fqkh3o4 {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Target Streamlit app header area */
    section[data-testid="stSidebar"] + div {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use a custom HTML title that's centered with no top margin/padding
st.markdown("<h1 style='margin-top:0; padding-top:0; text-align:center;'>NST+Upscaling custom solution by Helvin</h1>", unsafe_allow_html=True)

# Columns for content image upload + preview - use more balanced column sizing
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"], key="content")
with col2:
    if content_file:
        content_img = Image.open(content_file)
        st.image(content_img, caption="Content Image Preview", width=400)
    else:
        pass

# Add some space between sections
st.markdown("---")

# Columns for style image upload + preview
col3, col4 = st.columns([2, 1])
with col3:
    st.subheader("Style Image")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"], key="style")
with col4:
    if style_file:
        style_img = Image.open(style_file)
        st.image(style_img, caption="Style Image Preview", width=400)
    else:
        pass

# Move Process Images section outside of col4 and center it
if content_file and style_file:
    st.markdown("---")
    
    # Add custom CSS for centered header
    st.markdown("""
    <div style="text-align: center;">
    <h3>Process Images</h3>
    </div>
    """, unsafe_allow_html=True)

    # Add sliders for style transfer parameters
    num_steps = st.slider("Number of Optimization Steps", min_value=100, max_value=1000, value=500, step=50)
    style_weight = st.number_input("Style Weight", min_value=1e4, max_value=1e8, value=1e7, format="%e")
    content_weight = st.number_input("Content Weight", min_value=1, max_value=100, value=1, step=1)

    # Create columns to center the button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        process_button = st.button("Stylize and Upscale", type="primary", use_container_width=True)
    
    if process_button:
        timestamp = int(time.time())
        content_path = os.path.join(INPUT_DIR, f"content_{timestamp}.jpg")
        style_path = os.path.join(INPUT_DIR, f"style_{timestamp}.jpg")

        # Save uploaded files only once inside button click
        with open(content_path, "wb") as f:
            f.write(content_file.getbuffer())
        with open(style_path, "wb") as f:
            f.write(style_file.getbuffer())

        # Open uploaded images directly as PIL Images
        content = Image.open(content_path)
        style = Image.open(style_path)
        
        # Create a status container
        status_container = st.empty()
        status_container.info("Starting style transfer process...")
        
        with st.spinner("Running Style Transfer..."):
            with open("debug_log.txt", "a") as log_file:
                log_file.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"DEBUG: type(content) = {type(content)}\n")
                log_file.write(f"DEBUG: type(style) = {type(style)}\n")
                log_file.write(f"DEBUG: num_steps = {num_steps}\n")
                log_file.write(f"DEBUG: style_weight = {style_weight}\n")
                log_file.write(f"DEBUG: content_weight = {content_weight}\n")
            c_img = load_image(content, size=content.size[::-1])  # Use original size (width, height)
            s_img = load_image(style)
            output = run_style_transfer(c_img, s_img, num_steps=num_steps, style_weight=style_weight, content_weight=content_weight)
            result_img = im_convert(output)
            result_filename = f"styled_{timestamp}.jpg"
            result_path = os.path.join(STYLED_DIR, result_filename)
            result_img.save(result_path)
            status_container.success("Style transfer complete!")
            
            # Debug: log absolute path of saved styled image
            abs_result_path = os.path.abspath(result_path)
            with open("debug_log.txt", "a") as log_file:
                log_file.write(f"Styled image saved at absolute path: {abs_result_path}\n")
            
            # Debug: list files after saving stylized image
            stylized_files = os.listdir(STYLED_DIR)
            with open("debug_log.txt", "a") as log_file:
                log_file.write(f"Files in styled dir after saving stylized image: {stylized_files}\n")
        
        # Use full-width display for results
        st.markdown("## Results")
        st.markdown("### Stylized Image")
        st.image(result_img, caption="Stylized Result", width=400)
        
        status_container = st.empty()
        status_container.info("Starting upscaling process...")
        
        with st.spinner("Upscaling using Real-ESRGAN..."):
                upscaled_filename = f"upscaled_{timestamp}.jpg"
                upscaled_path = os.path.join(UPSCALED_DIR, upscaled_filename)
                # Pass output directory to upscale_image
                generated_file = upscale_image(result_path, UPSCALED_DIR)
                
                # Debug: list files after upscaling
                upscaled_files = os.listdir(UPSCALED_DIR)
                with open("debug_log.txt", "a") as log_file:
                    log_file.write(f"Files in upscaled dir after upscaling: {upscaled_files}\n")
                
                if generated_file is not None:
                    # Use the generated file path directly
                    upscaled_img = Image.open(generated_file)
                else:
                    st.error("Upscaling failed: no output file generated.")
                    status_container.error("Upscaling failed.")
                    upscaled_img = None
                
                if upscaled_img:
                    # Debug: log size of original and upscaled images
                    with open("debug_log.txt", "a") as log_file:
                        log_file.write(f"Original styled image size: {result_img.size}\n")
                        log_file.write(f"Upscaled image size: {upscaled_img.size}\n")
                    
                    status_container.success("Upscaling complete!")
                    
                    st.markdown("### Upscaled Image")
                    st.image(upscaled_img, caption="Upscaled Result", width=600)
        
        # Use a success message container with more details
        st.success(f"Processing complete! Images saved to {STYLED_DIR} and {UPSCALED_DIR}")
        st.info(f"Filenames: Style: {result_filename}, Upscaled: {upscaled_filename}")
else:
    # Show a message when not all images are uploaded
    if not (content_file and style_file):
        st.warning("Please upload both content and style images to proceed")

# Add spacing before footer and ensure it's visible
#st.markdown("<div style='margin-bottom:100px;'></div>", unsafe_allow_html=True)

# Custom footer implementation as fallback if the CSS approach doesn't work
st.markdown("""
<div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:30px;margin-top:0px; text-align:center;">
    <p style="color:black; margin:0;">© HMG</p>
</div>
""", unsafe_allow_html=True)
