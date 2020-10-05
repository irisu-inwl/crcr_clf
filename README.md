# crcr_clf
This is a machine learning web application that recognizes crcr images using streamlit and keras.  
- README_ja: [here](README_ja.md)
- [video demo](https://twitter.com/irisuinwl/status/1298259707602202625)

### Environment

- Windows10: Anaconda

### Configuration

Open Anaconda Prompt and do the following:

```
conda create -n keras python=3
conda activate keras
conda install tensorflow-gpu keras numpy pandas scikit-learn
pip install streamlit
```

### Getting Started

- Store a training model in the `model` directory.
- Run the following command with Anaconda Prompt

```
streamlit run app.py
```

### Deployment Using Docker

- building docker image

```
$ docker build -t crcrclf:latest .
```

- create container

```
$ docker run -id -p 8501:8501 --name crcrclf-test crcrclf:latest
```
