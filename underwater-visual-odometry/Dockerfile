# Use ROS Noetic as base image
FROM osrf/ros:noetic-desktop-full

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-tf \
    ros-noetic-rosbag \
    ros-noetic-rospy \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib==3.6.3 \
    tensorflow-cpu==2.10.0 \
    scikit-learn==1.2.2 \
    tqdm \
    pyquaternion \
    opencv-python==4.7.0.72 \
    Pillow==9.5.0 \
    scipy==1.10.1 \
    tensorboard==2.10.0 \
    h5py==3.8.0

# Create directories for project
RUN mkdir -p /app/data/raw /app/data/processed \
             /app/output/models /app/output/logs /app/output/visualizations \
             /app/scripts/models \
             /app/notebooks

# Copy your code into the container
COPY scripts/ /app/scripts/

# Set the working directory
WORKDIR /app

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
export CUDA_VISIBLE_DEVICES=""\n\
export PYTHONPATH="${PYTHONPATH}:/app"\n\
exec "$@"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]