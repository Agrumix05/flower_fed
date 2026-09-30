

"""quickstart-xgboost: A Flower / XGBoost app (Client)."""

import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from quickstart_xgboost.task import (
    build_xgb_params,
    compute_regression_metrics,
    load_client_data,
)

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    client_id = context.node_config["partition-id"]
    data_path = Path(context.run_config["data-path"]) / str(client_id)

    print(f"[CLIENT {client_id}] loading {data_path}")

    train_data, _, num_train, _ = load_client_data(data_path, train_frac=0.9, holdout_frac=0.1, exclude_prefixes=["s_", "d_"])

    cfg = unflatten_dict(context.run_config)
    params = build_xgb_params(cfg)
    local_rounds = context.run_config["local-epochs"]

    arrays = msg.content.get("arrays")
    if arrays is None or len(arrays) == 0 or len(arrays["0"].numpy()) == 0:
        booster = None
    else:
        booster = xgb.Booster(params=params)
        booster.load_model(bytearray(arrays["0"].numpy().tobytes()))

    model = xgb.train(
        params=params,
        dtrain=train_data,
        num_boost_round=local_rounds,
        xgb_model=booster,
    )

    model_bytes = model.save_raw(raw_format="json")
    model_array = np.frombuffer(model_bytes, dtype=np.uint8)

    train_metrics = compute_regression_metrics(
        train_data.get_label(), model.predict(train_data)
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord([model_array]),
                "metrics": MetricRecord(
                    {
                        "train_mae": train_metrics["mae"],
                        "num-examples": num_train,
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    client_id = context.node_config["partition-id"]
    data_path = Path(context.run_config["data-path"]) / str(client_id)

    train_data, valid_data, _, num_val = load_client_data(data_path, train_frac=0.9, holdout_frac=0.1,  exclude_prefixes=["s_", "d_"])
    # train_data, _, num_train, _ = load_client_data(data_path, train_frac=0.9, holdout_frac=0.1, exclude_prefixes=["s_", "d_"])


    booster = xgb.Booster()
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    booster.load_model(global_model)

    train_metrics = compute_regression_metrics(
        train_data.get_label(), booster.predict(train_data)
    )

    preds = booster.predict(valid_data)
    print(f"[CLIENT {client_id}] pred range: {preds.min():.2f} {preds.max():.2f}")

    val_metrics = compute_regression_metrics(valid_data.get_label(), preds)

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "train_mae": train_metrics["mae"],
                        "train_rmse": train_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                        "val_rmse": val_metrics["rmse"],
                        "num-examples": num_val,
                    }
                )
            }
        ),
        reply_to=msg,
    )