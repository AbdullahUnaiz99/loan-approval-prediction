import joblib
from pathlib import Path

def save_object(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_object(path):
    return joblib.load(path)
