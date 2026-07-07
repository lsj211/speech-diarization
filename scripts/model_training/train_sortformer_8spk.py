"""Exploratory fine-tuning script for expanding Sortformer to 8 speakers.

This is intentionally conservative for a course-project experiment:

* load a 4-speaker Sortformer checkpoint/model;
* expand the speaker output layers to ``--max-speakers``;
* train on chunked AISHELL-4 manifests;
* bypass the original PIL/ATS all-permutation target search, which becomes
  impractical at 8 speakers (8! permutations).

The direct-ATS loss assumes RTTM lines are sorted by time, so speaker columns
follow first-arrival order in the dataset target builder.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEMO_ROOT = PROJECT_ROOT / "baseline" / "NeMo"
torch = None
nn = None
pl = None
ModelCheckpoint = None
TensorBoardLogger = None
OmegaConf = None


def import_training_deps() -> None:
    global torch, nn, pl, ModelCheckpoint, TensorBoardLogger, OmegaConf

    import torch as torch_module
    import torch.nn as nn_module
    import lightning.pytorch as pl_module
    from lightning.pytorch.callbacks import ModelCheckpoint as model_checkpoint_cls
    from lightning.pytorch.loggers import TensorBoardLogger as tensorboard_logger_cls
    from omegaconf import OmegaConf as omegaconf_cls

    torch = torch_module
    nn = nn_module
    pl = pl_module
    ModelCheckpoint = model_checkpoint_cls
    TensorBoardLogger = tensorboard_logger_cls
    OmegaConf = omegaconf_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Sortformer with an expanded speaker output head.")
    parser.add_argument("--model-path", default="nvidia/diar_sortformer_4spk-v1")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_training/sortformer_8spk"))
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--session-len-sec", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=1.0,
        help="Weight applied to active-speaker frames in the direct BCE loss.",
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-name", default="sortformer_8spk_aishell4_train_L.nemo")
    return parser.parse_args()


def import_nemo_model():
    import sys

    nemo_root = str(NEMO_ROOT)
    if nemo_root not in sys.path:
        sys.path.insert(0, nemo_root)

    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel


def load_sortformer(model_path: str, model_cls):
    if model_path.endswith(".nemo"):
        return model_cls.restore_from(restore_path=model_path, map_location="cpu")
    if model_path.endswith(".ckpt"):
        return model_cls.load_from_checkpoint(checkpoint_path=model_path, map_location="cpu", strict=False)
    try:
        return model_cls.from_pretrained(model_path, map_location="cpu")
    except TypeError:
        return model_cls.from_pretrained(model_path)


def expand_linear(old_layer: nn.Linear, out_features: int) -> nn.Linear:
    if old_layer.out_features == out_features:
        return old_layer

    new_layer = nn.Linear(
        old_layer.in_features,
        out_features,
        bias=old_layer.bias is not None,
        device=old_layer.weight.device,
        dtype=old_layer.weight.dtype,
    )
    nn.init.xavier_uniform_(new_layer.weight)
    if new_layer.bias is not None:
        nn.init.zeros_(new_layer.bias)

    rows_to_copy = min(old_layer.out_features, out_features)
    with torch.no_grad():
        new_layer.weight[:rows_to_copy].copy_(old_layer.weight[:rows_to_copy])
        if old_layer.bias is not None:
            new_layer.bias[:rows_to_copy].copy_(old_layer.bias[:rows_to_copy])
    return new_layer


def expand_speaker_capacity(model, max_speakers: int) -> None:
    sortformer_modules = model.sortformer_modules
    sortformer_modules.n_spk = max_speakers
    sortformer_modules.single_hidden_to_spks = expand_linear(sortformer_modules.single_hidden_to_spks, max_speakers)
    sortformer_modules.hidden_to_spks = expand_linear(sortformer_modules.hidden_to_spks, max_speakers)

    OmegaConf.set_struct(model._cfg, False)
    model._cfg.max_num_of_spks = max_speakers
    model._cfg.sortformer_modules.num_spks = max_speakers
    OmegaConf.set_struct(model._cfg, True)

    # Prevent accidental construction/use of 8! permutation tables in this exploratory path.
    model.speaker_permutations = torch.empty(0, max_speakers, dtype=torch.long)


def masked_direct_bce_loss(preds, targets, target_lens, positive_class_weight: float):
    eps = 1e-6
    with torch.amp.autocast(device_type=preds.device.type, enabled=False):
        preds = preds.float().clamp(min=eps, max=1.0 - eps)
        targets = targets.to(device=preds.device, dtype=torch.float32)
        frame_idx = torch.arange(preds.shape[1], device=preds.device).unsqueeze(0)
        valid_mask = (frame_idx < target_lens.to(preds.device).unsqueeze(1)).unsqueeze(-1).to(preds.dtype)
        if positive_class_weight != 1.0:
            class_weight = torch.where(targets > 0.5, positive_class_weight, 1.0).to(preds.dtype)
        else:
            class_weight = 1.0
        loss = torch.nn.functional.binary_cross_entropy(preds, targets, reduction="none")
        loss = loss * valid_mask * class_weight
        return loss.sum() / valid_mask.expand_as(loss).sum().clamp_min(1.0)


def patch_direct_ats_loss(model, positive_class_weight: float) -> None:
    def _direct_train_eval(self, preds, targets, target_lens):
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])

        loss = masked_direct_bce_loss(preds, targets, target_lens, positive_class_weight)
        self._accuracy_train(preds, targets, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()
        learning_rate = self._optimizer.param_groups[0]["lr"] if self._optimizer is not None else 0.0
        return {
            "loss": loss,
            "ats_loss": loss.detach(),
            "pil_loss": torch.zeros_like(loss.detach()),
            "learning_rate": learning_rate,
            "train_f1_acc": train_f1_acc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1_acc_ats": train_f1_acc,
        }

    def _direct_val_eval(self, preds, targets, target_lens):
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])

        loss = masked_direct_bce_loss(preds, targets, target_lens, positive_class_weight)
        self._accuracy_valid(preds, targets, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()
        return {
            "val_loss": loss,
            "val_ats_loss": loss.detach(),
            "val_pil_loss": torch.zeros_like(loss.detach()),
            "val_f1_acc": val_f1_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_acc_ats": val_f1_acc,
        }

    model._get_aux_train_evaluations = types.MethodType(_direct_train_eval, model)
    model._get_aux_validation_evaluations = types.MethodType(_direct_val_eval, model)


def freeze_frontend_encoder(model) -> None:
    for module in (model.preprocessor, model.encoder):
        for param in module.parameters():
            param.requires_grad = False


def make_ds_config(args: argparse.Namespace, manifest: Path, is_train: bool) -> dict:
    return {
        "manifest_filepath": str(manifest.resolve()).replace("\\", "/"),
        "sample_rate": args.sample_rate,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "soft_label_thres": 0.5,
        "session_len_sec": args.session_len_sec,
        "num_spks": args.max_speakers,
        "soft_targets": False,
        "use_lhotse": False,
        "drop_last": False if not is_train else True,
    }


def configure_model(args: argparse.Namespace):
    SortformerEncLabelModel = import_nemo_model()
    model = load_sortformer(args.model_path, SortformerEncLabelModel)
    expand_speaker_capacity(model, args.max_speakers)
    patch_direct_ats_loss(model, args.positive_class_weight)

    if args.freeze_encoder:
        freeze_frontend_encoder(model)

    OmegaConf.set_struct(model._cfg, False)
    model._cfg.train_ds = make_ds_config(args, args.train_manifest, is_train=True)
    model._cfg.validation_ds = make_ds_config(args, args.val_manifest, is_train=False)
    model._cfg.optim = {
        "name": "adamw",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    OmegaConf.set_struct(model._cfg, True)
    return model


def main() -> None:
    args = parse_args()
    import_training_deps()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = configure_model(args)
    logger = TensorBoardLogger(save_dir=str(args.output_dir), name="tb")
    checkpoint = ModelCheckpoint(
        dirpath=str(args.output_dir / "checkpoints"),
        filename="epoch{epoch:02d}-valloss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=str(args.output_dir),
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=10,
    )
    model.set_trainer(trainer)
    model.setup_training_data(model._cfg.train_ds)
    model.setup_validation_data(model._cfg.validation_ds)

    trainer.fit(model)

    if trainer.is_global_zero:
        save_path = args.output_dir / args.save_name
        model.save_to(str(save_path))
        print(f"Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    main()
