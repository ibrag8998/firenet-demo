FROM python:3.8

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=off

RUN apt-get update && apt-get install python-opencv -y

WORKDIR /app/

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uwsgi", "./uwsgi.ini"]
