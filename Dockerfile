# Dockerfile

FROM python:3.10-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Set environment variable inside the container
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/GOOGLE_APPLICATION_CREDENTIALS

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]


