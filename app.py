from PIL import Image
import json
import os

import streamlit as st
import pandas as pd
import numpy as np

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

st.set_option('deprecation.showfileUploaderEncoding', False)

IS_WINDOWS = (os.name == 'nt')
if IS_WINDOWS:
    import keras.backend.tensorflow_backend as tb

    tb._SYMBOLIC_SCOPE.value = True
    

@st.cache(allow_output_mutation=True)
def load_config(config_path: str):
    """
    設定ファイルを読み込む
    """

    with open(config_path, 'r') as fr:
        config = json.load(fr)

    return config

@st.cache(allow_output_mutation=True)
def load_inception_v3(weight_path: str):
    """
    指定パスからパラメータ重みを読込、InceptionV3をfine-tuningしたモデルをloadする。
    """

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.load_weights(weight_path)

    return model

def preprocessing_image(image_pil_array: 'PIL.Image'):
    """
    予測するためにPIL.Imageで読み込んだarrayを加工する。
    """

    image_pil_array = image_pil_array.convert('RGB')
    image_pil_array = image_pil_array.resize((299,299))
    x = image.img_to_array(image_pil_array)

    x = np.expand_dims(x, axis=0)
    test_datagen = ImageDataGenerator(rescale=1./255)

    return test_datagen.flow(x)

config = load_config('config/config.json')
model_weight_path = config['model_weight_path']
class_indices = config['class_indices']
class_info = {v:k for k,v in class_indices.items()}

model = load_inception_v3(model_weight_path)

st.title('CRCR Classification')

st.write("""
クリクリ分類器です。  
アップロードしたクリクリの画像を分類します。  
画像ラベルがクリクリしかなく、異常検知とかしてないのでネギトロ丼がロージアちゃんになったりします。  
""")

uploaded_file = st.file_uploader('Choose a image file')

if uploaded_file is not None:
    image_pil_array = Image.open(uploaded_file)
    st.image(
        image_pil_array, caption='uploaded image',
        use_column_width=True
    )

    x = preprocessing_image(image_pil_array)
    result = model.predict_generator(x)

    predict_rank = np.argsort(np.ravel(result))[::-1]
    st.write('機械学習モデルは画像を', class_info[predict_rank[0]], 'ちゃんと予測しました。')

    st.write('#### 予測確率')
    df = pd.DataFrame(result.T, index=['holmy', 'jacklyn', 'rosia', 'tsukino'], columns=['predict_proba'])
    st.write(df)
    st.bar_chart(df)


