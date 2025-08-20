import argparse, cv2, numpy as np, tensorflow as tf, os

def load_img(path, sz=(80,80)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Cannot read image: "+path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, sz)
    img = img.astype('float32') / 255.0
    return img

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/mechanical_defect_model.h5")
    args = p.parse_args()
    model = tf.keras.models.load_model(args.model)
    img = load_img(args.image)
    x = np.expand_dims(img, 0)
    pred = model.predict(x)[0][0]
    label = "Defective" if pred >= 0.5 else "Non-Defective"
    conf = pred if pred>=0.5 else 1-pred
    print(f"{label} ({conf*100:.2f}% confidence)")
