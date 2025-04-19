FROM python:3.12.3
RUN mkdir /experiment
VOLUME "davide.domini-volume"
WORKDIR /experiment
COPY requirements.txt /experiment
RUN pip install -r requirements.txt
COPY src/ /experiment
COPY tests/ /experiment
ENV OWNER=1000:1000
CMD python3 tests/test_go_to_position.py