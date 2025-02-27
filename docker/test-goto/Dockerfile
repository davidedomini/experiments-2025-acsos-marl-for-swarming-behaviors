FROM python:3.12.3
RUN mkdir /experiment
WORKDIR /experiment
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "tests/test_go_to_position.py"]