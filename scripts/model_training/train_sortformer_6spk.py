"""Fine-tune Sortformer with the original ATS/PIL objective for 6 speakers.

This script keeps NeMo's original permutation-based training path and only
expands the pretrained 4-speaker Sortformer output head to 6 speakers. It is
intended as a more faithful, bounded-cost alternative to the exploratory 8spk
direct-BCE script.
"""

from __future__ import annotations

import argparse
import itertools
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
    parser = argparse.ArgumentParser(description="Fine-tune Sortformer 6spk with original ATS/PIL loss.")
    parser.add_argument("--model-path", default="models/diar_sortformer_4spk-v1.nemo")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/model_training/sortformer_6spk_original"))
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--session-len-sec", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pil-weight", type=float, default=None, help="Override model PIL loss weight.")
    parser.add_argument("--ats-weight", type=float, default=None, help="Override model ATS loss weight.")
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Number of best validation checkpoints to keep.",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=4,
        help="Save an additional periodic checkpoint every N epochs. Use 0 to disable.",
    )
    parser.add_argument("--save-name", default="sortformer_6spk_original_aishell4_train_LM.nemo")
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


def refresh_speaker_permutations(model, max_speakers: int) -> None:
    speaker_inds = list(range(max_speakers))
    model.speaker_permutations = torch.tensor(
        list(itertools.permutations(speaker_inds)),
        dtype=torch.long,
    )
    print(f"Speaker permutations: {model.speaker_permutations.shape[0]} for {max_speakers} speakers")


def expand_speaker_capacity(model, max_speakers: int) -> None:
    sortformer_modules = model.sortformer_modules
    sortformer_modules.n_spk = max_speakers
    sortformer_modules.single_hidden_to_spks = expand_linear(sortformer_modules.single_hidden_to_spks, max_speakers)
    sortformer_modules.hidden_to_spks = expand_linear(sortformer_modules.hidden_to_spks, max_speakers)

    OmegaConf.set_struct(model._cfg, False)
    model._cfg.max_num_of_spks = max_speakers
    model._cfg.sortformer_modules.num_spks = max_speakers
    OmegaConf.set_struct(model._cfg, True)
    refresh_speaker_permutations(model, max_speakers)


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
    if args.pil_weight is not None:
        model._cfg.pil_weight = args.pil_weight
    if args.ats_weight is not None:
        model._cfg.ats_weight = args.ats_weight
    OmegaConf.set_struct(model._cfg, True)
    model._init_loss_weights()
    return model


def main() -> None:
    args = parse_args()
    import_training_deps()
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = configure_model(args)
    logger = TensorBoardLogger(save_dir=str(args.output_dir), name="tb")
    callbacks = []
    best_checkpoint = ModelCheckpoint(
        dirpath=str(args.output_dir / "checkpoints"),
        filename="best-epoch{epoch:02d}-valloss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(best_checkpoint)
    if args.save_every_n_epochs > 0:
        periodic_checkpoint = ModelCheckpoint(
            dirpath=str(args.output_dir / "checkpoints"),
            filename="periodic-epoch{epoch:02d}",
            every_n_epochs=args.save_every_n_epochs,
            save_top_k=-1,
            save_last=False,
            auto_insert_metric_name=False,
        )
        callbacks.append(periodic_checkpoint)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=str(args.output_dir),
        callbacks=callbacks,
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
