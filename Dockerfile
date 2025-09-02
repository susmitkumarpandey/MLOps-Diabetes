# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt if you have dependencies
COPY requirements.txt .
COPY app/ ./app/
COPY diabetes.csv .
COPY train.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python train.py

# Expose the port your app runs on (change if needed)
EXPOSE 8080

# Command to run your application (update as needed)
CMD ["python", "app/app.py"]