from tensorflow.keras.models import load_model
import os
import gdown
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import requests
import io

# Загрузка модели
def load_colorization_model(model_url, model_path='model.h5'):
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    model = load_model(model_path)
    return model

# Загрузка черно-белого изображения из URL
def load_grayscale_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).convert('L')
    return np.array(image, dtype=np.float32)

# Предобработка черно-белого изображения
def preprocess_grayscale_image(grayscale_image):
    original_size = grayscale_image.shape[:2]
    grayscale_image_resized = Image.fromarray(grayscale_image.astype(np.uint8)).resize((256, 256), Image.BILINEAR)
    img_gray_arr = np.array(grayscale_image_resized) / 255.0
    img_lab = rgb2lab(np.stack([img_gray_arr]*3, axis=-1))[:, :, 0]
    img_lab = img_lab.reshape(1, img_lab.shape[0], img_lab.shape[1], 1)
    return img_lab, original_size

# Генерация окрашенного изображения
def generate_colorized_image(model, img_lab, original_size):
    output = model.predict(img_lab)
    output *= 128
    output = np.clip(output[0], -128, 127)
    result_img = np.zeros((img_lab.shape[1], img_lab.shape[2], 3))
    result_img[:, :, 0] = img_lab[0, :, :, 0]
    result_img[:, :, 1:] = output
    colorized_image = lab2rgb(result_img)
    colorized_image_pil = Image.fromarray((colorized_image * 255).astype(np.uint8))
    colorized_image_resized = colorized_image_pil.resize((original_size[1], original_size[0]), Image.BILINEAR)
    return colorized_image_resized

# Функция для запуска остальных
def main():
    model_url = 'https://drive.google.com/uc?id=1--sZIzfnDlm4F7EZtSs7QXhPACOdSO2-'
    model = load_colorization_model(model_url)

    grayscale_image_url = input("Введите URL черно-белого изображения: ")
    try:
        grayscale_image = load_grayscale_image_from_url(grayscale_image_url)
        img_lab, original_size = preprocess_grayscale_image(grayscale_image)
        colorized_image = generate_colorized_image(model, img_lab, original_size)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(grayscale_image, cmap='gray')
        plt.title('Input Grayscale Image')

        plt.subplot(1, 2, 2)
        plt.imshow(colorized_image)
        plt.title('Generated Colorized Image')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Произошла ошибка:", e)

if __name__ == "__main__":
    main()