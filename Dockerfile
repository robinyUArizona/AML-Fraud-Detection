# Use the official python image from the Docker Hub 
FROM python:3.12-slim-bookworm
# Set the working directory
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Ensures that the system's package list is up-to-date and then installs the AWS CLI on the system
RUN apt update -y && apt install awscli -y
# Install any needed packages specified in the requirements.txt
RUN pip install -r requirements.txt
# Run app.py when the container launches
# CMD ["python3", "app.py"]
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]