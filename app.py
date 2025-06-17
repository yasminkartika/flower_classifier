import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model yang sudah dilatih
model = tf.keras.models.load_model('best_model.h5')

# Daftar kelas bunga (ganti jika berbeda)
class_names = ['Lily', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Sesuaikan dengan input model
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Tambahkan batch dimensi
    img_array = img_array / 255.0  # Normalisasi piksel
    return img_array

# Judul aplikasi
st.title("🌸 Klasifikasi Gambar Bunga dengan CNN MobileNetV2")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar bunga (jpg/jpeg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    st.write("🔍 Memprediksi...")
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"🌼 Prediksi: **{predicted_class}**")
    st.info(f"📊 Akurasi keyakinan: **{confidence*100:.2f}%**")
