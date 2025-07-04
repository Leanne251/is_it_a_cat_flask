# Use Python base image
FROM python:3.10


WORKDIR /app

COPY . /app
# Install pipreqs and generate a clean one
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Expose default port
EXPOSE 7860

# Run with Gunicorn
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=7860" ]