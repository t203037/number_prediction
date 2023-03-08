import streamlit as st
from streamlit_drawable_canvas import st_canvas
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')

from deep import model
#アプリ作成

# モデルをロードする
model = load_model('my_model.h5')

# スケッチパッドを定義する
SIZE = 192
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode='freedraw',
    key='canvas')

# スケッチパッドから数字を予測する関数を定義する
@st.cache(allow_output_mutation=True)
def predict_digit(canvas):
    # スケッチパッドの描画結果をnumpy配列に変換する
    img = canvas.image_data.astype('float32') / 255.0
    img = np.expand_dims(img[:,:,0], axis=-1)
    #画像サイズ変更
    img = Image.fromarray((img.squeeze() * 255).astype(np.uint8)).resize((28, 28))
    img=np.array(img).astype('float32')/255.0
    img=np.expand_dims(img,axis=0)
    # 予測を行う
    pred = model.predict(img)
    return pred

# スケッチパッドを表示し、数字を予測するボタンを追加する
if canvas_result.image_data is not None:
    if st.button('Predict'):
        pred = predict_digit(canvas_result)
        pred_label = np.argmax(pred)
        st.write('Predicted Label:', pred_label)
        st.bar_chart(pred.reshape(10))
