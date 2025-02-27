FROM python:3.12.3
RUN mkdir /experiment
WORKDIR /experiment
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/training/train_gcn_dqn.py"]