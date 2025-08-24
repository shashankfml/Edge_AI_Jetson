import streamlit as st
import cv2
import pandas as pd
import os
import io
import hashlib
from datetime import datetime
from pathlib import Path
from app.core.detection import VehicleProcessor

# Configuration
PROCESS_VIDEO_DIR = Path("process_video")
PROCESS_VIDEO_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Vehicle Detection", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
process_every = st.sidebar.slider("Process Every N Frames", 1, 10, 3)

def get_file_hash(file_bytes):
    """Generate hash for file content to detect duplicates"""
    return hashlib.md5(file_bytes).hexdigest()

def main():
    st.title("🚗 Automated Vehicle Detection System")
    
    # Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        file_hash = get_file_hash(uploaded_file.getvalue())
        
        # Check if this file was already processed
        if file_hash in st.session_state.processed_files:
            st.info("This video was already processed. Showing previous results.")
            st.session_state.current_results = st.session_state.processed_files[file_hash]
        else:
            # Save to process_video folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = PROCESS_VIDEO_DIR / f"input_{timestamp}_{uploaded_file.name}"
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the video
            with st.spinner("Processing video..."):
                processor = VehicleProcessor()
                results = processor.process_video(
                    input_path=str(video_path),
                    display_frame=st.empty(),
                    progress_bar=st.progress(0),
                    confidence_thresh=confidence_thresh,
                    process_every=process_every
                )
                
                # Store results
                st.session_state.current_results = results
                st.session_state.processed_files[file_hash] = results
            
            # Clean up video file
            try:
                os.remove(video_path)
            except Exception as e:
                st.warning(f"Could not remove video file: {e}")
            
            st.success("Processing complete!")
    
    # Display results if available
    if st.session_state.current_results:
        st.dataframe(st.session_state.current_results['dataframe'])
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download CSV",
                data=st.session_state.current_results['csv_data'],
                file_name="vehicle_results.csv",
                mime="text/csv"
            )
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                st.session_state.current_results['dataframe'].to_excel(writer, index=False)
            
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name="vehicle_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Option to clear results
        if st.button("Clear Results"):
            st.session_state.current_results = None
            st.rerun()

if __name__ == "__main__":
    main()