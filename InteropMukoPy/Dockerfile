FROM python:3.6


WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD [ "python", "./opt/pht_train/main.py"]