FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip

RUN pip install \
    matplotlib==3.8.0 \
    numpy==1.26.4 \
    openpyxl==3.1.2 \
    pandas==1.4.4 \
    python_mnist==0.7 \
    scikit-learn==1.4.0 \
    scipy==1.12.0

RUN pip install \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install \
    jupyterlab \
    notebook \
    ipykernel

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]