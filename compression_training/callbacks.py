from __future__ import annotations

import os
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import TrainerCallback

__all__ = [
    "PreCreateCkptDirCallback",
    "ParamChangeTrackerCallback",
    "EarlyStoppingStatusCallback",
    "PrintMetricsCallback",
]


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_global_rank0() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _is_local_process_zero(state) -> bool:
    if hasattr(state, "is_local_process_zero"):
        return bool(state.is_local_process_zero)
    return _is_global_rank0()


class PreCreateCkptDirCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        save_strategy = getattr(args, "save_strategy", None)
        save_steps = int(getattr(args, "save_steps", 0) or 0)
        if str(save_strategy) != "steps" or save_steps <= 0:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or step % save_steps != 0:
            return

        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        if _is_dist_initialized():
            if _is_global_rank0():
                os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()
        else:
            os.makedirs(ckpt_dir, exist_ok=True)


class ParamChangeTrackerCallback(TrainerCallback):
    def __init__(
        self,
        model: nn.Module,
        track_patterns: Optional[Sequence[str]] = None,
        topn: int = 10,
        skip_first_log: bool = True,
        verbose_on_bind: bool = True,
    ):
        super().__init__()
        self.model = model
        self.track_patterns = list(track_patterns) if track_patterns is not None else [
            "h2e",
            "student_lm.model.layers.",
            "student_lm.lm_head",
            "lat_bos",
        ]
        self.topn = int(topn)
        self.skip_first_log = bool(skip_first_log)
        self.verbose_on_bind = bool(verbose_on_bind)

        self._bound = False
        self._tracked: List[Tuple[str, nn.Parameter]] = []
        self._init_snap: Dict[str, torch.Tensor] = {}
        self._prev_snap: Dict[str, torch.Tensor] = {}
        self._step_grad_norm: Dict[str, float] = {}
        self._did_first_log = False

    def _select_tracked(self) -> List[Tuple[str, nn.Parameter]]:
        tracked = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(pattern in name for pattern in self.track_patterns):
                tracked.append((name, param))
        return tracked

    def _take_snapshot(self) -> None:
        self._init_snap = {name: param.detach().float().cpu().clone() for name, param in self._tracked}
        self._prev_snap = {name: param.detach().float().cpu().clone() for name, param in self._tracked}

    def on_train_begin(self, args, state, control, **kwargs):
        self._tracked = self._select_tracked()
        self._take_snapshot()
        self._step_grad_norm = {}
        self._bound = True

        if not _is_local_process_zero(state):
            return
        if not self._tracked:
            print("[ParamChangeTracker] WARNING: no parameters matched", self.track_patterns)
            return
        if self.verbose_on_bind:
            print("[ParamChangeTracker] Tracking parameters:")
            for name, _ in self._tracked:
                print("  -", name)

    def on_step_end(self, args, state, control, **kwargs):
        if not self._bound:
            return
        self._step_grad_norm = {}
        for name, param in self._tracked:
            if param.grad is not None:
                self._step_grad_norm[name] = float(param.grad.detach().float().norm().item())

    def _compute_stats(self) -> List[Tuple[str, float, float, float, float]]:
        stats = []
        for name, param in self._tracked:
            cur = param.detach().float().cpu()
            theta_norm = float(cur.norm().item())
            delta_prev = float((cur - self._prev_snap[name]).norm().item())
            delta_init = float((cur - self._init_snap[name]).norm().item())
            grad_norm = float(self._step_grad_norm.get(name, 0.0))
            stats.append((name, theta_norm, delta_prev, delta_init, grad_norm))
            self._prev_snap[name] = cur.clone()
        stats.sort(key=lambda item: item[2], reverse=True)
        return stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._bound:
            return
        if self.skip_first_log and not self._did_first_log:
            self._did_first_log = True
            return
        if not _is_local_process_zero(state) or not self._tracked:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        print(f"[param-delta] step={step} (top {self.topn})")
        print(f"{'name':60s}  {'d_prev':>10s}  {'d_init':>10s}  {'theta':>10s}  {'grad':>10s}")
        for name, theta_norm, delta_prev, delta_init, grad_norm in self._compute_stats()[: self.topn]:
            print(f"{name:60s}  {delta_prev:10.4e}  {delta_init:10.4e}  {theta_norm:10.4e}  {grad_norm:10.4e}")


class EarlyStoppingStatusCallback(TrainerCallback):
    def __init__(
        self,
        metric_for_best: str,
        greater_is_better: bool,
        patience: int,
        threshold: float,
        show_last: int = 5,
    ):
        self.metric_for_best = str(metric_for_best)
        self.greater_is_better = bool(greater_is_better)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.show_last = int(show_last)
        self.best: Optional[float] = None
        self.bad_count = 0
        self.history: Deque[Tuple[int, float, bool]] = deque(maxlen=max(5, self.show_last))

    def _metric_key(self, metrics: Dict[str, float]) -> str:
        key = self.metric_for_best
        if not key.startswith("eval_"):
            key = f"eval_{key}"
        return key if key in metrics else "eval_loss"

    def _is_improved(self, cur: float) -> bool:
        if self.best is None:
            return True
        if self.greater_is_better:
            return cur > self.best + self.threshold
        return cur < self.best - self.threshold

    def on_train_begin(self, args, state, control, **kwargs):
        if _is_local_process_zero(state):
            direction = "higher" if self.greater_is_better else "lower"
            print(
                f"[early-stop] watching {self.metric_for_best}, target={direction}, "
                f"patience={self.patience}, threshold={self.threshold}"
            )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not _is_local_process_zero(state) or not isinstance(metrics, dict):
            return

        key = self._metric_key(metrics)
        if key not in metrics:
            print(f"[early-stop] metric {key!r} not found. available: {list(metrics.keys())}")
            return

        cur = float(metrics[key])
        step = int(getattr(state, "global_step", 0) or 0)
        improved = self._is_improved(cur)
        if improved:
            self.best = cur
            self.bad_count = 0
        else:
            self.bad_count += 1

        self.history.append((step, cur, improved))
        remaining = max(0, self.patience - self.bad_count)
        print(
            f"[early-stop] step={step} {key}={cur:.6f} | best={self.best:.6f} | "
            f"improved={improved} | patience={self.bad_count}/{self.patience} | remaining={remaining}"
        )


class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        if hasattr(model, "module"):
            model = model.module

        metrics = getattr(model, "last_metrics", None)
        if not metrics:
            return

        logging_steps = int(getattr(args, "logging_steps", 1) or 1)
        step = int(getattr(state, "global_step", 0) or 0)
        if step % logging_steps != 0 or not _is_local_process_zero(state):
            return

        print(
            f"[step {step}] "
            f"ce_theta={float(metrics.get('ce_theta', -1)):.6f} "
            f"kl={float(metrics.get('kl', -1)):.6f} "
            f"align={float(metrics.get('align', -1)):.6f} "
            f"total={float(metrics.get('total', -1)):.6f}"
        )
