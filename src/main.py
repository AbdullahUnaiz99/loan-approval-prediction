import argparse
import pandas as pd
from pathlib import Path
import yaml
from model_training import train_model
from data_preprocessing import create_preprocessor

CONFIG = yaml.safe_load(Path("config/config.yaml").read_text())

def main(stage, data_path):
    df = pd.read_csv(data_path)
    if stage in {"all", "train"}:
        train_model(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=["train", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--data", required=True, help="Path to raw CSV"
    )
    args = parser.parse_args()
    main(args.stage, args.data)
