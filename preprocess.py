"""
preprocess.py
- Simple helper: convert folder-structured data into TFRecord (optional).
"""
import tensorflow as tf
from pathlib import Path
import os

def build_tfrecords(data_dir, out_file):
    writer = tf.io.TFRecordWriter(out_file)
    class_map = {'non_defective':0, 'defective':1}
    for cls in class_map:
        p = Path(data_dir) / cls
        if not p.exists():
            continue
        for f in p.glob('*.*'):
            img = open(f, 'rb').read()
            feature = {
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_map[cls]]))
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(ex.SerializeToString())
    writer.close()
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <data_dir> <out.tfrecord>")
    else:
        build_tfrecords(sys.argv[1], sys.argv[2])
