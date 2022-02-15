FROM python:3.6

LABEL envs="[\"FHIR_SERVER\",\"FHIR_PORT\"]"

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD [ "python", "./main.py" ]