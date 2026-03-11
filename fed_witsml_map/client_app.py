"""Fed-WITSML-Map: Flower ClientApp.

Each client represents a different service company or operator with its own
mnemonic naming conventions.  Clients train a local character-level classifier
and share only model weight updates — never raw mnemonic mapping tables.
"""

import copy

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from flwr.clientapp import ClientApp
from flwr.common import Context, Message
from flwr.common.record import ArrayRecord, MetricRecord, RecordDict

from .task import (
    evaluate_model,
    get_model,
    load_demo_data,
    load_sim_data,
    train_fn,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the mnemonic mapper on local vendor/operator data."""
    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    proximal_mu = float(msg.content["config"].get("proximal-mu", 0.0))
    global_params = list(copy.deepcopy(model).parameters()) if proximal_mu > 0 else None

    cfg = context.run_config
    batch_size = int(cfg.get("batch-size", 32))
    local_epochs = int(cfg.get("local-epochs", 2))
    lr = float(cfg.get("learning-rate", 0.001))
    test_fraction = float(cfg.get("test-fraction", 0.2))
    seed = int(cfg.get("seed", 42))
    samples_per_class = int(cfg.get("samples-per-class", 40))

    is_sim = "partition-id" in context.node_config and "num-partitions" in context.node_config

    if is_sim:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, testloader = load_sim_data(
            partition_id, num_partitions, batch_size,
            test_fraction=test_fraction, seed=seed,
            samples_per_class=samples_per_class,
        )
    else:
        trainloader, testloader = load_demo_data(
            batch_size=batch_size, test_fraction=test_fraction, seed=seed,
        )

    train_loss = train_fn(
        model, trainloader, local_epochs, lr, device,
        valloader=testloader,
        proximal_mu=proximal_mu,
        global_params=global_params,
    )
    eval_loss, eval_acc, preds, labels = evaluate_model(model, testloader, device)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)

    metrics = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_f1_macro": f1,
        "eval_precision_macro": prec,
        "eval_recall_macro": rec,
        "num-examples": len(trainloader.dataset),
    }
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on local hold-out data."""
    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = context.run_config
    batch_size = int(cfg.get("batch-size", 32))
    test_fraction = float(cfg.get("test-fraction", 0.2))
    seed = int(cfg.get("seed", 42))
    samples_per_class = int(cfg.get("samples-per-class", 40))

    is_sim = "partition-id" in context.node_config and "num-partitions" in context.node_config

    if is_sim:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        _, testloader = load_sim_data(
            partition_id, num_partitions, batch_size,
            test_fraction=test_fraction, seed=seed,
            samples_per_class=samples_per_class,
        )
    else:
        _, testloader = load_demo_data(
            batch_size=batch_size, test_fraction=test_fraction, seed=seed,
        )

    eval_loss, eval_acc, preds, labels = evaluate_model(model, testloader, device)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)

    metrics = {
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_f1_macro": f1,
        "eval_precision_macro": prec,
        "eval_recall_macro": rec,
        "num-examples": len(testloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
