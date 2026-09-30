"""quickstart_xgboost: Flower / XGBoost Server."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbCyclic

from quickstart_xgboost.task import replace_keys










from pathlib import Path

import pandas as pd
from quickstart_xgboost.task import build_centralized_test_set, compute_regression_metrics

def evaluate_final_model(model: xgb.Booster, data_root: str):
    df = build_centralized_test_set(data_root)

    y = df["label"].to_numpy(dtype=np.float32)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    dtest = xgb.DMatrix(X, label=y)

    preds = model.predict(dtest)
    metrics = compute_regression_metrics(y, preds)

    print("\n" + "=" * 60)
    print("FINAL GLOBAL EVALUATION")
    print("=" * 60)
    print(f"Samples: {len(y)}")
    print(f"MAE    : {metrics['mae']:.6f}")
    print(f"RMSE   : {metrics['rmse']:.6f}")
    print("=" * 60)












app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:

    # ============================================================
    # CONFIG
    # ============================================================

    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    cfg = replace_keys(
        unflatten_dict(context.run_config)
    )

    params = cfg["params"]

    print("=" * 60)
    print("Flower XGBoost Federated Training (FedXgbCyclic)")
    print(f"Rounds           : {num_rounds}")
    print(f"Train fraction   : {fraction_train}")
    print(f"Evaluate fraction: {fraction_evaluate}")
    print("=" * 60)

    # ============================================================
    # INITIAL MODEL
    # ============================================================

    initial_arrays = ArrayRecord(
        [
            np.frombuffer(
                b"",
                dtype=np.uint8,
            )
        ]
    )

    # ============================================================
    # STRATEGY
    # ============================================================

    strategy = FedXgbCyclic(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        weighted_by_key="num-examples",
    )

    # ============================================================
    # TRAINING
    # ============================================================

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # ============================================================
    # FINAL MODEL
    # ============================================================

    model_bytes = bytearray(
        result.arrays["0"]
        .numpy()
        .tobytes()
    )

    print()
    print("=" * 60)
    print("FINAL MODEL")
    print("=" * 60)
    print(f"Model size : {len(model_bytes)/1024:.2f} KB")

    booster = xgb.Booster(params=params)
    booster.load_model(model_bytes)

    print(
        f"Boosted rounds: {booster.num_boosted_rounds()}"
    )

    output_path = Path("final_model.json")

    booster.save_model(output_path)



    print(f"Model saved to: {output_path.resolve()}")

    print("=" * 60)
    print("Training completed.")
    print("=" * 60)

    evaluate_final_model(
        booster,
        context.run_config["data-path"],
    )