"""Fed-WITSML-Map: Flower ServerApp.

Uses FedProx to handle the non-IID mnemonic distributions across vendors.
Each service company uses different naming conventions for the same physical
measurements — FedProx's proximal term prevents any single vendor's conventions
from dominating the global model during local training.

No Differential Privacy is used: mnemonic mappings (e.g., "GAM" -> gamma_ray)
are not sensitive data.  The FL value here is convenience, not secrecy — operators
contribute their mapping knowledge passively without maintaining a shared database.
"""

from pathlib import Path

import torch

from flwr.app import ConfigRecord
from flwr.common import Context
from flwr.common.record import ArrayRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

from .task import evaluate_model, get_model, load_sim_data

app = ServerApp()


def _global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    context: Context,
) -> dict:
    """Server-side evaluation on partition 0's test data."""
    try:
        model = get_model()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _, testloader = load_sim_data(
            partition_id=0,
            num_partitions=max(1, int(context.run_config.get("num-partitions", 5))),
            batch_size=int(context.run_config.get("batch-size", 32)),
            samples_per_class=int(context.run_config.get("samples-per-class", 80)),
        )
        loss, acc, _, _ = evaluate_model(model, testloader, device)
        return {"eval_loss": loss, "eval_accuracy": acc}
    except Exception:
        return {}


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedProx for WITSML mnemonic mapping."""
    cfg = context.run_config
    num_rounds = int(cfg.get("num-server-rounds", 10))
    lr = float(cfg.get("learning-rate", 0.003))
    fraction_evaluate = float(cfg.get("fraction-evaluate", 1.0))
    proximal_mu = float(cfg.get("proximal-mu", 0.1))

    global_model = get_model()
    arrays = ArrayRecord(global_model.state_dict())

    def evaluate_fn(server_round: int, arr: ArrayRecord) -> dict:
        return _global_evaluate(server_round, arr, context)

    strategy = FedProx(
        proximal_mu=proximal_mu,
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    out_path = Path(cfg.get("model-save-path", "output/witsml_mapper.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.arrays.to_torch_state_dict(), out_path)
    print(f"\nSaved global WITSML mapper to {out_path}")
