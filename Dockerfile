# # Use official Python image
# FROM python:3.9-slim-bookworm

# # Install OS-level dependencies
# # Install system dependencies with build tools
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app_project

# # Copy only necessary files
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --upgrade pip && pip install -r requirements.txt

# # Copy application code
# COPY . .

# # Create necessary directories
# RUN mkdir -p /app_project/feedback/images /app_project/feedback/labels /app_project/process_video

# # Environment variables
# ENV PYTHONPATH=/app_project

# # Expose Streamlit default port
# EXPOSE 8501

# # Run the Streamlit app
# CMD ["streamlit", "run", "final_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Use official Python image
FROM python:3.9-slim-bookworm

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app_project

# First copy only requirements for caching
COPY requirements.txt .

# Install Python dependencies with no-cache-dir
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app_project/feedback/images \
    /app_project/feedback/labels \
    /app_project/process_video

# Environment variables
ENV PYTHONPATH=/app_project

# Expose Streamlit default port
EXPOSE 8501


# Run the Streamlit app
CMD ["streamlit", "run", "final_app.py", "--server.port=8501", "--server.address=0.0.0.0"]