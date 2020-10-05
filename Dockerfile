FROM python:latest

ADD . /opt/crcrclf
WORKDIR /opt/crcrclf

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# ここらへんでgit cloneとか機械学習モデルの読込とか出来たらgood

CMD streamlit run app.py