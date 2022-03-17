FROM python:3.7

WORKDIR /usr/src/app/


## Install all libraries
COPY requirement.txt .
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirement.txt

## Copy Source
COPY . .

#Add App to PYTHON Path
ENV PYTHONPATH="/usr/src/app:${PATH}"

## Run
CMD [ "python", "./opt/pht_train/main.py"]
