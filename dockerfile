FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose the port Render expects
EXPOSE 5000

# Use Gunicorn with 1 worker to save RAM
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
