import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from backend.config import DIGIT_MODEL_PATH, SAVED_MODELS_DIR
from backend.models import _create_digit_model


def train_digit_model():
    print('=' * 50)
    print('MNIST 数字识别模型训练 (轻量增强)')
    print('=' * 50)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    print(f'训练: {x_train.shape[0]} | 测试: {x_test.shape[0]}')

    datagen = ImageDataGenerator(
        rotation_range=12, width_shift_range=0.12,
        height_shift_range=0.12, shear_range=0.1,
        zoom_range=0.12, fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = _create_digit_model()

    model.fit(
        datagen.flow(x_train, y_train, batch_size=128, shuffle=True),
        epochs=25,
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=3,
                min_lr=1e-6, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=6,
                restore_best_weights=True, verbose=1
            )
        ],
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'测试准确率: {acc * 100:.2f}%')

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model.save(DIGIT_MODEL_PATH)
    print(f'模型已保存: {DIGIT_MODEL_PATH}')


if __name__ == '__main__':
    train_digit_model()
