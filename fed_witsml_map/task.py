"""Fed-WITSML-Map: model, tokeniser, data loading, and training utilities.

The model is a character-level 1D-CNN that classifies WITSML mnemonic strings
(plus their unit-of-measure) into standard PWLS property classes.  Character-
level processing means the model can generalise to mnemonics it has never seen
before — it learns patterns like "GR" ⊂ "ECGR" ⊂ "HSGR" all map to gamma_ray.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .mnemonic_catalog import (
    NUM_CLASSES,
    generate_vendor_data,
)


# ---------------------------------------------------------------------------
# Character tokeniser
# ---------------------------------------------------------------------------

_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/. "
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(_CHARS)}  # 0 = PAD
CHAR_VOCAB_SIZE = len(CHAR_TO_IDX) + 2  # +1 PAD, +1 UNK

MAX_MNEM_LEN = 24
MAX_UNIT_LEN = 12


def _tokenise(text: str, max_len: int) -> list[int]:
    """Convert a string to a list of character token IDs, padded to max_len."""
    text = text.upper()
    ids = [CHAR_TO_IDX.get(c, len(CHAR_TO_IDX) + 1) for c in text[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


def tokenise_batch(
    mnemonics: list[str],
    units: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise lists of mnemonic and unit strings into integer tensors."""
    m = torch.tensor([_tokenise(m, MAX_MNEM_LEN) for m in mnemonics], dtype=torch.long)
    u = torch.tensor([_tokenise(u, MAX_UNIT_LEN) for u in units], dtype=torch.long)
    return m, u


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MnemonicMapper(nn.Module):
    """Character-level 1D-CNN for WITSML mnemonic classification.

    Takes tokenised mnemonic and unit strings and classifies them into one
    of NUM_CLASSES standard PWLS property classes.

    Architecture:
        mnemonic chars → embedding → Conv1d → Conv1d → AdaptiveAvgPool
        unit chars     → embedding → Conv1d → AdaptiveAvgPool
        concat → Linear → ReLU → Dropout → Linear → logits
    """

    def __init__(
        self,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mnem_embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.mnem_conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([hidden_dim, MAX_MNEM_LEN]),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.unit_embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.unit_conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        mnem_ids: torch.Tensor,
        unit_ids: torch.Tensor,
    ) -> torch.Tensor:
        m = self.mnem_embed(mnem_ids).transpose(1, 2)
        m = self.mnem_conv(m).squeeze(-1)

        u = self.unit_embed(unit_ids).transpose(1, 2)
        u = self.unit_conv(u).squeeze(-1)

        return self.classifier(torch.cat([m, u], dim=1))


def get_model(num_classes: int = NUM_CLASSES) -> MnemonicMapper:
    """Return a new MnemonicMapper instance."""
    return MnemonicMapper(num_classes=num_classes)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_loaders(
    data: list[tuple[str, str, int]],
    batch_size: int,
    test_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Convert raw (mnemonic, unit, label) tuples into train/test DataLoaders."""
    mnems = [d[0] for d in data]
    units = [d[1] for d in data]
    labels = [d[2] for d in data]

    m_ids, u_ids = tokenise_batch(mnems, units)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(y)
    n_test = max(1, int(n * test_fraction))
    n_train = n - n_test

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(m_ids[train_idx], u_ids[train_idx], y[train_idx])
    test_ds = TensorDataset(m_ids[test_idx], u_ids[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    test_fraction: float = 0.2,
    seed: int = 42,
    samples_per_class: int = 40,
) -> tuple[DataLoader, DataLoader]:
    """Generate per-vendor simulation data for a federated client.

    Each partition simulates a different service company or operator with
    distinct mnemonic naming conventions.
    """
    data = generate_vendor_data(
        vendor_id=partition_id,
        num_vendors=num_partitions,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    return _build_loaders(data, batch_size, test_fraction, seed + partition_id)


def load_demo_data(
    batch_size: int = 32,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Generate a single-node demo dataset for Deployment mode.

    In production, replace this with a loader that reads the operator's
    internal mnemonic mapping table (CSV with mnemonic, unit, property columns).
    """
    return load_sim_data(
        partition_id=0,
        num_partitions=1,
        batch_size=batch_size,
        test_fraction=test_fraction,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_fn(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    valloader: Optional[DataLoader] = None,
    weight_decay: float = 1e-5,
    proximal_mu: float = 0.0,
    global_params: Optional[list] = None,
) -> float:
    """Train the model for one federated round.

    Uses class-weighted CrossEntropyLoss and optional FedProx proximal term.
    """
    net.to(device)
    net.train()

    all_labels = torch.cat([ys for _, _, ys in trainloader]).long()
    class_counts = torch.bincount(all_labels, minlength=NUM_CLASSES).float().clamp(min=1)
    class_weights = (1.0 / class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss = 0.0
    n_batches = 0
    best_state = None
    best_val_loss = float("inf")

    for _epoch in range(epochs):
        for m_ids, u_ids, ys in trainloader:
            m_ids = m_ids.to(device)
            u_ids = u_ids.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            loss = criterion(net(m_ids, u_ids), ys)

            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = sum(
                    (lp - gp.to(device)).norm(2)
                    for lp, gp in zip(net.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        if valloader is not None:
            val_loss, _, _, _ = evaluate_model(net, valloader, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(net.state_dict())

    if best_state is not None:
        net.load_state_dict(best_state)

    return running_loss / n_batches if n_batches else 0.0


def evaluate_model(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate the model. Returns (loss, accuracy, all_preds, all_labels)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for m_ids, u_ids, ys in testloader:
            m_ids = m_ids.to(device)
            u_ids = u_ids.to(device)
            ys = ys.to(device)
            out = net(m_ids, u_ids)
            total_loss += criterion(out, ys).item()
            pred = out.argmax(dim=1)
            correct += (pred == ys).sum().item()
            total += ys.size(0)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(ys.cpu().tolist())

    loss = total_loss / len(testloader) if testloader else 0.0
    accuracy = correct / total if total else 0.0
    return loss, accuracy, all_preds, all_labels
