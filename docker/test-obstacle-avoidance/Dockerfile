FROM python:3.12.3
RUN mkdir /experiment
VOLUME "davide.domini-volume"
WORKDIR /experiment
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV OWNER=1000:1000
CMD python3 tests/test_obstacle_avoidance.py