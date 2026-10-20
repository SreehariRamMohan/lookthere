"""Train and evaluate LookThere on one or more GPUs.

    # 1) Warm up the extractor on all patches.
    python train.py warmup configs/billiards/billiards_warmup.yaml
    # 2) Train the selector (GRPO) and the extractor (cross-entropy) together from the warmup checkpoint.
    python train.py rl configs/billiards/billiards_95_rl.yaml init=runs/warmup_max-left-right/warmup.pt
    # 3) Evaluate any checkpoint.
    python train.py eval configs/billiards/billiards_95_rl.yaml ckpt=runs/rl_billiards_95/final.pt

Any config key can be overridden on the command line as key=value. For several GPUs, launch with
`torchrun --nproc_per_node=N train.py ...`. Each GPU then trains on its own `batch_size` images.
"""

import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data import DATASETS
from data.augment import CutMix
from objectives import make_objective
from modeling import Extractor, Selector, mask_to_padded_indices

torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------------------------
# Multiple GPUs
# ---------------------------------------------------------------------------------------------
# Each GPU runs its own copy of the models on its own batch, and the gradients are averaged before
# every optimizer step. With one GPU every function below does nothing.

def setup_distributed():
    """Join the process group when launched by torchrun with several GPUs. Returns the rank."""
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return 0
    dist.init_process_group("nccl", timeout=timedelta(hours=2))  # long enough for a full evaluation
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank()


def distributed():
    return dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def is_main():
    return rank() == 0


def barrier():
    if distributed():
        dist.barrier()


def broadcast_parameters(*modules):
    """Copy the weights of GPU 0 to the other GPUs (random initializations differ between GPUs)."""
    if distributed():
        for module in modules:
            for tensor in module.state_dict().values():
                dist.broadcast(tensor, src=0)


def average_gradients(module):
    """Average the gradients over GPUs, so that every GPU takes the same optimizer step."""
    if distributed():
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def info(message):
    if is_main():
        print(message)


# ---------------------------------------------------------------------------------------------
# Config, seeding, data
# ---------------------------------------------------------------------------------------------

def _coerce(value):
    """YAML reads '5e-5' as a string. Turn strings that look like numbers into numbers."""
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return SimpleNamespace(**{k: _coerce(v) for k, v in cfg.items()})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    base = torch.initial_seed() % (2 ** 32)
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


def make_val_loader(cfg, split="valid"):
    """Download the data if needed and return the loader of an evaluation split.

    With several GPUs each GPU evaluates its own share of the split (see `evaluate`).
    """
    dataset = DATASETS[cfg.dataset]
    if is_main():
        dataset.prepare(cfg.data_root)  # one GPU downloads, the others wait
    barrier()
    data = dataset.load(cfg, split)
    sampler = DistributedSampler(data, shuffle=False) if distributed() else None
    return DataLoader(data, batch_size=cfg.batch_size, shuffle=False, sampler=sampler,
                      num_workers=cfg.num_workers, drop_last=False)


def make_loaders(cfg):
    val_loader = make_val_loader(cfg)
    dataset = DATASETS[cfg.dataset]
    # CutMix and label smoothing make the training labels soft targets (see data.augment).
    mix = cfg.cutmix_alpha > 0 or cfg.label_smoothing > 0
    collate_fn = CutMix(dataset.NUM_CLASSES, cfg.cutmix_alpha, cfg.label_smoothing) if mix else None
    seeded = cfg.seed is not None
    data = dataset.load(cfg, "train")
    sampler = DistributedSampler(data, shuffle=True, seed=cfg.seed or 0, drop_last=True) if distributed() else None
    train_loader = DataLoader(
        data, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=cfg.num_workers,
        collate_fn=collate_fn, drop_last=True, prefetch_factor=4, persistent_workers=True,
        worker_init_fn=_worker_init_fn if seeded else None,
        generator=torch.Generator().manual_seed(cfg.seed + rank()) if seeded else None,
    )
    return train_loader, val_loader


def set_epoch(loader, epoch):
    """With several GPUs, reshuffle how the training set is split between the GPUs."""
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


# ---------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------

def cosine_lr(step, max_lr, min_lr, total_steps):
    """Warm up linearly over the first 10% of the steps. Then decay to min_lr with a half cosine."""
    warmup_steps = math.ceil(total_steps * 0.1)
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return min_lr + (max_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))
    )


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def active_levels(rl_step, grid_stages, steps_per_stage):
    """Grid curriculum. Level k joins the cascade after the first k stages are over."""
    end = 0
    for i, steps in enumerate(steps_per_stage):
        end += steps
        if rl_step < end:
            return grid_stages[:i + 1]
    return list(grid_stages)


def policy_schedule(cfg, progress):
    """Return the temperature (linear) and the entropy coefficient (V-shaped) at an RL progress in [0, 1]."""
    t_init, t_final = cfg.temperature
    temperature = t_init + progress * (t_final - t_init)
    e_init, e_mid, e_final = cfg.entropy_coeff
    if progress < 0.5:
        entropy_coeff = e_init + progress / 0.5 * (e_mid - e_init)
    else:
        entropy_coeff = e_mid + (progress - 0.5) / 0.5 * (e_final - e_mid)
    return temperature, entropy_coeff


# ---------------------------------------------------------------------------------------------
# Reward and GRPO
# ---------------------------------------------------------------------------------------------

def sparsity_reward(keep_mask, target):
    """Reward for dropping patches. It depends on the kept fraction f of the mask.

    The reward peaks at the keep rate `target`. It falls off linearly as 1 - d with
    d = |f - target| / max(target, 1 - target).
    """
    frac = keep_mask.float().mean(dim=-1)
    return 1.0 - (frac - target).abs() / max(target, 1.0 - target)


def current_target(cfg, progress):
    """The keep rate target at an RL progress in [0, 1]. A list [start, end] is a linear ramp."""
    target = cfg.sparsity_target
    return target[0] + progress * (target[1] - target[0]) if isinstance(target, list) else target


def mask_reward(extractor, objective, images, targets, prefix, keep_mask, sparsity_target, cfg):
    """Reward of one sampled mask: -loss + lambda * gate * sparsity_reward. It has no gradients."""
    keep_idx, pad = mask_to_padded_indices(keep_mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = extractor(images, prefix.detach(), keep_idx, pad)
        loss = objective.loss(extractor, out, targets, keep_idx, pad)
    sparsity = sparsity_reward(keep_mask, sparsity_target)
    return -loss + cfg.lam * (sparsity * objective.gate(extractor, out, targets))


# ---------------------------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------------------------

def warmup_step(selector, extractor, objective, opt_ext, images, labels, step, cfg):
    """Train the extractor on all patches. In the second half some patches are dropped at random."""
    opt_ext.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        targets = objective.targets(images, labels)
        with torch.no_grad():
            _, prefix = selector(images)
        keep_mask = torch.ones(images.shape[0], selector.grid ** 2, dtype=torch.bool, device=images.device)
        progress = step / cfg.schedule_steps
        if cfg.patch_drop > 0 and progress >= cfg.patch_drop_start:
            ramp = (progress - cfg.patch_drop_start) / (1.0 - cfg.patch_drop_start)
            drop_prob = cfg.patch_drop * ramp
            keep_mask = (torch.rand_like(keep_mask.float()) > drop_prob).to(keep_mask.dtype)
        keep_idx, pad = mask_to_padded_indices(keep_mask)
        out = extractor(images, prefix, keep_idx, pad)
        loss = objective.loss(extractor, out, targets, keep_idx, pad)
    loss.mean().backward()
    average_gradients(extractor)
    torch.nn.utils.clip_grad_norm_(extractor.clip_params(), 1.0)
    opt_ext.step()
    return {"loss": loss.mean().item(), "k_mean": keep_mask.float().sum(dim=-1).mean().item()}


def rl_step(selector, extractor, objective, opt_sel, opt_ext, images, labels, levels, temperature, entropy_coeff,
            sparsity_target, cfg):
    """One joint update. The extractor learns from a sampled mask and the selector learns with GRPO."""
    opt_ext.zero_grad()
    opt_sel.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        targets = objective.targets(images, labels)
        logits, prefix = selector(images)
        keep_mask, log_probs, entropy = selector.sample(logits, levels, temperature)
        keep_idx, pad = mask_to_padded_indices(keep_mask)
        out = extractor(images, prefix.detach(), keep_idx, pad)
        loss = objective.loss(extractor, out, targets, keep_idx, pad)

    # Extractor: supervised update on the first sampled mask.
    loss.mean().backward()
    average_gradients(extractor)
    torch.nn.utils.clip_grad_norm_(extractor.clip_params(), 1.0)
    opt_ext.step()

    # Selector: GRPO over M masks of the same image. In the paper's runs the extractor step above
    # comes first. So the updated extractor scores masks 2 to M. The gate of the first mask (the
    # correctness for classification) also uses the updated head. We keep that order.
    with torch.no_grad():
        gate = objective.gate(extractor, out, targets)
    sparsity = sparsity_reward(keep_mask, sparsity_target)
    rewards = [-loss.detach() + cfg.lam * (sparsity * gate)]
    sum_log_probs = [log_probs.sum(dim=-1)]
    for _ in range(cfg.num_samples - 1):
        mask_m, log_probs_m, _ = selector.sample(logits, levels, temperature)
        with torch.no_grad():
            rewards.append(mask_reward(extractor, objective, images, targets, prefix, mask_m, sparsity_target, cfg))
        sum_log_probs.append(log_probs_m.sum(dim=-1))
    rewards = torch.stack(rewards)              # (M, bs)
    sum_log_probs = torch.stack(sum_log_probs)  # (M, bs)

    group_std = rewards.std(dim=0, keepdim=True).clamp(min=cfg.min_group_std)
    advantages = ((rewards - rewards.mean(dim=0, keepdim=True)) / group_std)
    advantages = advantages.clamp(-cfg.advantage_clip, cfg.advantage_clip)
    policy_loss = -(sum_log_probs * advantages.detach()).mean()
    num_actions = sum(g * g for g in levels)
    entropy_bonus = -entropy_coeff * (entropy.mean() / num_actions)
    (policy_loss + entropy_bonus).backward()
    average_gradients(selector)
    torch.nn.utils.clip_grad_norm_(selector.parameters(), 0.5)
    opt_sel.step()

    metrics = {"loss": loss.mean().item()}
    if objective.metric == "accuracy":
        metrics["train_acc"] = gate.mean().item()
    metrics.update({
        "k_mean": keep_mask.float().sum(dim=-1).mean().item(),
        "sparsity": 1.0 - keep_mask.float().mean().item(),
        "reward": rewards.mean().item(),
        "policy_loss": policy_loss.item(),
        "entropy": entropy.mean().item(),
    })
    return metrics


# ---------------------------------------------------------------------------------------------
# Evaluation and checkpoints
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(selector, extractor, objective, loader, levels=None, temperature=None, deterministic=False):
    """Measure the metrics of the objective and the sparsity on `loader` with masks sampled from the selector.

    All patches are kept if `levels` is None. The metrics are means of per-batch means, as in the
    paper's runs. With several GPUs the batches of all GPUs are pooled.
    """
    selector.eval()
    extractor.eval()
    totals, num_batches = {}, 0
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            targets = objective.targets(images, labels)
            logits, prefix = selector(images)
            if levels is None:
                keep_mask = torch.ones_like(logits, dtype=torch.bool)
            else:
                keep_mask, _, _ = selector.sample(logits, levels, temperature, deterministic)
            keep_idx, pad = mask_to_padded_indices(keep_mask)
            out = extractor(images, prefix, keep_idx, pad)
            batch = objective.eval_metrics(extractor, out, targets, keep_idx, pad)
        batch = {k: v.float().mean().item() for k, v in batch.items()}
        batch["sparsity"] = (1.0 - keep_mask.float().mean(dim=-1)).mean().item()
        batch["k_mean"] = keep_mask.float().sum(dim=-1).mean().item()
        for k, v in batch.items():
            totals[k] = totals.get(k, 0.0) + v
        num_batches += 1
    selector.train()
    extractor.train()
    if distributed():
        keys = sorted(totals)
        summed = torch.tensor([totals[k] for k in keys] + [num_batches], dtype=torch.float64, device="cuda")
        dist.all_reduce(summed)
        totals, num_batches = dict(zip(keys, summed[:-1].tolist())), summed[-1].item()
    return {k: v / num_batches for k, v in totals.items()}


def save_checkpoint(path, selector, extractor, opt_ext, opt_sel, phase, step, rl_step, temperature, cfg):
    if not is_main():  # the weights are the same on every GPU
        return
    torch.save({
        "selector": selector.state_dict(),
        "extractor": extractor.state_dict(),
        "opt_extractor": opt_ext.state_dict(),
        "opt_selector": opt_sel.state_dict() if opt_sel is not None else None,
        "phase": phase,
        "step": step,          # Global step (warmup and RL). It drives the RL learning rate schedule.
        "rl_step": rl_step,
        "temperature": temperature,
        "logit_offset": selector.logit_offset,
        "config": vars(cfg),
    }, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"Saved {path}")


def build_models(cfg, pretrained=True):
    selector = Selector(cfg.img_size, cfg.grid_stages, cfg.beta_init, cfg.beta_max, pretrained, cfg.logit_offset).cuda()
    extractor = Extractor(cfg.img_size, DATASETS[cfg.dataset].NUM_CLASSES, pretrained).cuda()
    return selector, extractor


class Logger:
    """Write metrics to output_dir/log.jsonl (and W&B) from GPU 0. Training metrics are those of GPU 0's batch."""

    def __init__(self, cfg):
        self.enabled = is_main()
        if not self.enabled:
            return
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.file = open(os.path.join(cfg.output_dir, "log.jsonl"), "a")
        self.wandb = None
        if cfg.wandb_project:
            import wandb
            self.wandb = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    def log(self, metrics, step):
        if not self.enabled:
            return
        self.file.write(json.dumps({"step": step, **metrics}) + "\n")
        self.file.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)


def _fmt(metrics):
    return ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())


# ---------------------------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------------------------

def run_warmup(cfg):
    """Phase 1: train the extractor on all patches. The selector only supplies its prefix tokens."""
    selector, extractor = build_models(cfg, pretrained=cfg.resume is None)
    objective = make_objective(cfg)
    opt_ext = torch.optim.AdamW(extractor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    min_lr = cfg.min_lr if cfg.min_lr is not None else cfg.lr * 0.1
    train_loader, val_loader = make_loaders(cfg)
    logger = Logger(cfg)

    step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=True)
        selector.load_state_dict(ckpt["selector"])
        extractor.load_state_dict(ckpt["extractor"])
        opt_ext.load_state_dict(ckpt["opt_extractor"])
        step = ckpt["step"]
    broadcast_parameters(selector, extractor)

    epoch = 0
    while step < cfg.steps:
        set_epoch(train_loader, epoch)
        epoch += 1
        for images, labels in tqdm(train_loader, desc=f"warmup {step}/{cfg.steps}", disable=not is_main()):
            if step >= cfg.steps:
                break
            set_lr(opt_ext, cosine_lr(step, cfg.lr, min_lr, cfg.schedule_steps))
            metrics = warmup_step(selector, extractor, objective, opt_ext, images.cuda(), labels.cuda(), step, cfg)
            if step % cfg.log_every == 0:
                logger.log(metrics, step)
                info(f"step={step}, {_fmt(metrics)}")
            step += 1
            if step % cfg.save_every == 0 and step < cfg.steps:
                save_checkpoint(os.path.join(cfg.output_dir, f"warmup_step{step}.pt"),
                                selector, extractor, opt_ext, None, "warmup", step, 0, None, cfg)
        val = evaluate(selector, extractor, objective, val_loader)
        logger.log({f"val/{k}": v for k, v in val.items()}, step)
        info(f"[eval] step={step}, {_fmt(val)}")

    save_checkpoint(os.path.join(cfg.output_dir, "warmup.pt"),
                    selector, extractor, opt_ext, None, "warmup", step, 0, None, cfg)


def load_distilled(selector, extractor, ckpt):
    """Load the weights of a distillation checkpoint. The betas and the extractor head can differ."""
    no_head = {k: v for k, v in ckpt["extractor"].items() if not k.startswith("head.")}
    for module, state, ignore in ((selector, ckpt["selector"], "betas."), (extractor, no_head, "head.")):
        result = module.load_state_dict(state, strict=False)
        wrong = [k for k in result.missing_keys + result.unexpected_keys if not k.startswith(ignore)]
        if wrong:
            raise RuntimeError(f"Cannot load the keys {wrong} from the distillation checkpoint.")


@torch.no_grad()
def fit_linear_probe(selector, extractor, cfg):
    """Initialize the classification head with a logistic regression on frozen features, as in the paper.

    The features are the CLS tokens of the extractor with all patches kept, on the training images
    (with the training augmentation).
    """
    from sklearn.linear_model import LogisticRegression

    def features(split):
        data = DATASETS[cfg.dataset].load(cfg, split)
        loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        feats, labels = [], []
        for images, y in tqdm(loader, desc=f"linear probe: {split} features"):
            images = images.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, prefix = selector(images)
                keep_mask = torch.ones(images.shape[0], selector.grid ** 2, dtype=torch.bool, device=images.device)
                keep_idx, pad = mask_to_padded_indices(keep_mask)
                feats.append(extractor(images, prefix, keep_idx, pad)["cls"].float().cpu())
            labels.append(y)
        return torch.cat(feats).numpy(), torch.cat(labels).numpy()

    selector.eval()
    extractor.eval()
    train_x, train_y = features("train")
    val_x, val_y = features("valid")
    probe = LogisticRegression(max_iter=1000, C=0.316, solver="lbfgs", n_jobs=-1).fit(train_x, train_y)
    extractor.head.weight.copy_(torch.from_numpy(probe.coef_))
    extractor.head.bias.copy_(torch.from_numpy(probe.intercept_))
    selector.train()
    extractor.train()
    print(f"Linear probe: val accuracy {probe.score(val_x, val_y):.4f} with all patches")


def run_rl(cfg):
    """Phase 2: train the selector with GRPO and the extractor on the loss of the objective.

    Without `init` or `resume` it starts from pretrained DINOv2.
    """
    ckpt_path = cfg.resume or cfg.init
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True) if ckpt_path else None
    selector, extractor = build_models(cfg, pretrained=ckpt is None)
    objective = make_objective(cfg)
    opt_sel = torch.optim.AdamW(selector.parameters(), lr=cfg.selector_lr, weight_decay=cfg.weight_decay)
    opt_ext = torch.optim.AdamW(extractor.parameters(), lr=cfg.extractor_lr, weight_decay=cfg.weight_decay)

    # Start from a warmup checkpoint (weights and the AdamW state of the extractor), from a
    # distillation checkpoint (weights only) or resume an RL run.
    rl_step_idx, start_step = 0, 0
    if ckpt is not None and ckpt["phase"] == "distill" and not cfg.resume:
        load_distilled(selector, extractor, ckpt)
        start_step = ckpt["step"]
        if cfg.loss == "classification" and cfg.linear_probe and is_main():
            fit_linear_probe(selector, extractor, cfg)  # the other GPUs get the head from the broadcast below
    elif ckpt is not None:
        selector.load_state_dict(ckpt["selector"])
        extractor.load_state_dict(ckpt["extractor"])
        opt_ext.load_state_dict(ckpt["opt_extractor"])
        for group in opt_ext.param_groups:
            group["weight_decay"] = cfg.weight_decay
        if ckpt["phase"] == "warmup":
            start_step = ckpt["step"]
        else:
            opt_sel.load_state_dict(ckpt["opt_selector"])
            rl_step_idx, start_step = ckpt["rl_step"], ckpt["step"] - ckpt["rl_step"]
            selector.logit_offset = ckpt.get("logit_offset", cfg.logit_offset)
    broadcast_parameters(selector, extractor)
    phase = "distill" if cfg.loss == "distillation" else "rl"

    # The learning rates continue the step count of the warmup. There is one cosine schedule over
    # the warmup and RL steps and RL sees the tail of it, as in the paper's runs.
    rl_steps = sum(cfg.steps_per_stage)
    total_steps = start_step + rl_steps
    sel_min_lr = cfg.min_lr if cfg.min_lr is not None else cfg.selector_lr * 0.1
    ext_min_lr = cfg.min_lr if cfg.min_lr is not None else cfg.extractor_lr * 0.1
    train_loader, val_loader = make_loaders(cfg)
    logger = Logger(cfg)

    metric, short = objective.metric, objective.short_name
    best_score, best_val, temperature = float("-inf"), {}, cfg.temperature[0]
    epoch = 0
    while rl_step_idx < rl_steps:
        set_epoch(train_loader, epoch)
        epoch += 1
        for images, labels in tqdm(train_loader, desc=f"rl {rl_step_idx}/{rl_steps}", disable=not is_main()):
            if rl_step_idx >= rl_steps:
                break
            step = start_step + rl_step_idx
            progress = rl_step_idx / rl_steps
            levels = active_levels(rl_step_idx, cfg.grid_stages, cfg.steps_per_stage)
            temperature, entropy_coeff = policy_schedule(cfg, progress)
            sparsity_target = current_target(cfg, progress)
            set_lr(opt_sel, cosine_lr(step, cfg.selector_lr, sel_min_lr, total_steps))
            set_lr(opt_ext, cosine_lr(step, cfg.extractor_lr, ext_min_lr, total_steps))
            metrics = rl_step(selector, extractor, objective, opt_sel, opt_ext, images.cuda(), labels.cuda(),
                              levels, temperature, entropy_coeff, sparsity_target, cfg)
            if step % cfg.log_every == 0:
                metrics.update(temperature=temperature, entropy_coeff=entropy_coeff, sparsity_target=sparsity_target,
                               grid=levels[-1], **{f"beta_{i}": b.item() for i, b in enumerate(selector.betas)})
                logger.log(metrics, step)
                info(f"step={step}, {_fmt(metrics)}")
            rl_step_idx += 1
            if rl_step_idx % cfg.save_every == 0 and rl_step_idx < rl_steps:
                save_checkpoint(os.path.join(cfg.output_dir, f"rl_step{rl_step_idx}.pt"), selector, extractor,
                                opt_ext, opt_sel, phase, start_step + rl_step_idx, rl_step_idx, temperature, cfg)
        levels = active_levels(rl_step_idx, cfg.grid_stages, cfg.steps_per_stage)
        val = evaluate(selector, extractor, objective, val_loader, levels, temperature)
        if objective.sign * val[metric] > best_score:
            best_score, best_val = objective.sign * val[metric], val
        logger.log({**{f"val/{k}": v for k, v in val.items()}, f"val/best_{metric}": best_val[metric]},
                   start_step + rl_step_idx)
        info(f"[eval] rl_step={rl_step_idx}, {_fmt(val)}, best_{short}={best_val[metric]:.4f}")

    save_checkpoint(os.path.join(cfg.output_dir, "final.pt"), selector, extractor, opt_ext, opt_sel,
                    phase, start_step + rl_step_idx, rl_step_idx, temperature, cfg)
    final = evaluate(selector, extractor, objective, val_loader,
                     active_levels(rl_step_idx, cfg.grid_stages, cfg.steps_per_stage), temperature)
    best_val = best_val or final  # Nothing was evaluated if the run was already finished.
    summary = {f"final_val_{short}": final[metric], "final_sparsity": final["sparsity"],
               f"best_val_{short}": best_val[metric], f"sparsity_at_best_val_{short}": best_val["sparsity"]}
    if is_main():
        with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    info(f"Summary: {summary}")


def run_eval(cfg):
    selector, extractor = build_models(cfg, pretrained=False)
    ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
    selector.load_state_dict(ckpt["selector"])
    extractor.load_state_dict(ckpt["extractor"])
    selector.logit_offset = ckpt.get("logit_offset", cfg.logit_offset)  # the value the model was trained with
    objective = make_objective(cfg)
    val_loader = make_val_loader(cfg, cfg.split)
    if ckpt["phase"] == "warmup":
        metrics = evaluate(selector, extractor, objective, val_loader)
    else:
        levels = active_levels(ckpt["rl_step"], cfg.grid_stages, cfg.steps_per_stage)
        metrics = evaluate(selector, extractor, objective, val_loader, levels, ckpt["temperature"], cfg.deterministic)
    info(f"[eval] {cfg.ckpt} on {cfg.split}: {_fmt(metrics)}")


DEFAULTS = dict(dataset="billiards", loss="classification", augment="default", cutmix_alpha=0.0, label_smoothing=0.0,
                seed=0, resume=None, init=None, ckpt=None, split="valid", deterministic=False, min_lr=None, logit_offset=0.0,
                wandb_project=None, run_name=None, log_every=5, save_every=1000000, linear_probe=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["warmup", "rl", "eval"])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    for key, value in DEFAULTS.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    gpu = setup_distributed()
    if cfg.seed is not None:
        seed_everything(cfg.seed + gpu)  # different masks and augmentations on every GPU

    start = time.time()
    {"warmup": run_warmup, "rl": run_rl, "eval": run_eval}[args.phase](cfg)
    info(f"Done in {(time.time() - start) / 3600:.2f} h")
    if distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
