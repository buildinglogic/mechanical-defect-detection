import tensorflow as tf
from tensorflow.keras import layers
import os
from model import build_model

DATA_DIR = os.getenv('DATA_DIR', '/content/mech_data')  # change if needed
IMG_SIZE = (80,80)
BATCH = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='binary',
    validation_split=0.2, subset='training', seed=123,
    image_size=IMG_SIZE, batch_size=BATCH
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, labels='inferred', label_mode='binary',
    validation_split=0.2, subset='validation', seed=123,
    image_size=IMG_SIZE, batch_size=BATCH
)

resize_rescale = tf.keras.Sequential([layers.Rescaling(1./255)])
train_ds = train_ds.map(lambda x,y: (resize_rescale(x), y)).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x,y: (resize_rescale(x), y)).cache().prefetch(tf.data.AUTOTUNE)

model = build_model(input_shape=(80,80,3), efficient=False)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('models/mechanical_defect_model.h5', save_best_only=True, monitor='val_accuracy')
]

model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
print("Saved best model to models/mechanical_defect_model.h5")
