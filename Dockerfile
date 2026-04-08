FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Flask/Waitress will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
