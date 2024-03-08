FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

COPY . .

EXPOSE 5000
 
CMD ["gunicorn", "-b","0.0.0.0:7860","main:app"]
