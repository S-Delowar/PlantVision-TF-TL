# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements.txt from flask/ to /app
COPY requirements.txt . 

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app.py from flask/ to /app
COPY . . 


# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]



# Commands for building and running the app:
# docker build -t plantvision-flask-app .
# docker run -d -p 5000:5000 --name plantvision-flask-container plantvision-flask-app
