FROM anibali/pytorch:1.8.1-cuda11.1

WORKDIR / app/ -> WORKDIR /app

COPY requirements.txt requirements.txt

COPY setup.py setup.py

COPY src/ src/

COPY data/ data/

COPY models/ models/

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py", "data/processed", "models/model.pth"]
