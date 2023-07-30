# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Set the working directory to /workspace
WORKDIR /workspace

# Update package lists and install required packages
RUN apt-get update && apt-get install -y python3-tk ffmpeg libsm6 libxext6

# Copy the requirements.txt file to the container's /workspace
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install -r requirements.txt

# Set environment variable for X11 display
ENV DISPLAY=unix$DISPLAY

# Start a bash shell in interactive mode when the container is run
CMD ["python", "main.py"]

