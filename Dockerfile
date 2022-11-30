# From the source image
FROM python:3.10-alpine

# Identify maintainer
LABEL maintainer="pz108366@student.sgh.waw.pl"

WORKDIR /app/

# Copy requirements.txt to /app/ inside the container
COPY requirements.txt /app/

# Install required packages
RUN pip install -r ./requirements.txt

COPY binary_model.sav app.py /app/

EXPOSE 9696

ENTRYPOINT python ./app.py
