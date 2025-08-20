# Mechanical Part Defect Detection Using AI

Production-style crack detection pipeline for mechanical parts.
- Train a CNN (optionally EfficientNet) to classify parts as defective (crack) or non-defective.
- Includes data pipeline, training script, prediction script, Streamlit demo, Dockerfile, and deployment instructions.

Quick start (in Colab):
1. Train model (example): run `train.py` or the Colab notebook.
2. Save best model: `models/mechanical_defect_model.h5`
3. Deploy the app (Streamlit): `streamlit run app/streamlit_app.py`

See the `app/` folder for the demo, and `sample_small_dataset/` for a tiny dataset generator.

