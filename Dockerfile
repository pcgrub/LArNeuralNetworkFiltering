FROM tensorflow/tensorflow

WORKDIR /lar

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY models_src/ ./models_src

COPY Run_NN.py .

CMD [ "python3", "./Run_NN.py" ]