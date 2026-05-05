import os
import gdown

def download_models():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "predictor", "models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {
        "efficientnet_model.keras": "https://drive.google.com/uc?id=1kFXsRY9ChEAz2ky-3JSF6rxopDErYYxb"
    }

    for filename, url in models.items():
        output_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            gdown.download(url, output_path, quiet=False)
        else:
            print(f"{filename} already exists.")