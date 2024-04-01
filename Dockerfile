FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt /code/

# Uncomment and adjust if you need system dependencies
# RUN apt-get update && apt-get install -y \
#     default-libmysqlclient-dev \
#     build-essential \
#     pkg-config
RUN pip install pydantic==1.10.13
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

# Run the ingest script, but don't fail the build if it fails
RUN python /code/ingest.py || true

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
