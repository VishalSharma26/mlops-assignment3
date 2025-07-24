
FROM python:3.10-slim

WORKDIR /app

COPY train.py predict.py ./

RUN pip install scikit-learn joblib

RUN python train.py

CMD ["python", "predict.py"]
