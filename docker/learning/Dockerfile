FROM python:3.12.3
RUN mkdir /experiment
VOLUME "davide.domini-volume"
WORKDIR /experiment
COPY requirements.txt /experiment
RUN pip install -r requirements.txt
COPY . /experiment
ENV OWNER=1000:1000
CMD python3 src/training/train_gcn_dqn.py