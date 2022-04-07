FROM python:3

WORKDIR /unidata

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./unidata

CMD [ "python3", "subp.py" ]