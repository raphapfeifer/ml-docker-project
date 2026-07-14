import pickle
import argparse
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

def load_model(model_path):
    logger.info("Model loading...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_eff(model, x):
    x_array = np.array(x).reshape(1,-1)
    logger.info("Doing efficiency predict...")
    res = model.predict(x_array)
    logger.info(f"Efficiency predict at the day {x} : {res[0]}")

def predict_data(model, y):
    coef = model.coef_
    intercept = model.intercept_

    day = (y - intercept) / coef[0]

    logger.info(f"Predict with efficiency {y} | at the day: {day}")


if __name__ == "__main__":
    model = load_model('artifacts/heat_efficiency_model.pkl')
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--efficiency', type=int)
    group.add_argument('--date', type=float)
    args = parser.parse_args()

    if args.efficiency is not None:
        predict_eff(model, args.efficiency)
    elif args.date is not None:
        predict_data(model, args.date)    