"""
grpo_trainer.py
===============
Group Relative Policy Optimization (GRPO) for the TraceGuard Overseer.

Architecture change from supervised MLP
─────────────────────────────────────────
Old:  MLP(trace) → sigmoid scalar → threshold → action
New:  Policy(trace) → softmax(3) → action distribution → sample

GRPO vs PPO
────────────
PPO needs a separate value network V(s) to estimate baselines:
    advantage = R - V(s)

GRPO eliminates the value network by using *group-relative* baselines:
    For each trace, sample G completions (action sequences) from the policy.
    advantage_i = (R_i - mean(R_group)) / (std(R_group) + eps)

This is exactly what DeepSeek-R1 / Llama reasoning models use. For TraceGuard,
"one completion" = one overseer action on one trace.

Step-aware SFT loss
────────────────────
Different pipeline steps have different risk profiles (STEP_RISK_WEIGHTS).
The SFT cross-entropy loss is multiplied by the step's risk weight so the
policy learns to be *more decisive* at high-risk steps (deployer=1.5)
and slightly more lenient at low-risk steps (tester=0.8).

KL penalty
───────────
To prevent lexical overfitting (the key failure mode for TraceGuard — where
the policy memorizes keywords rather than learning semantic threat patterns),
GRPO adds a KL divergence penalty between the current policy and a frozen
reference policy:
    loss = -GRPO_surrogate + kl_coef * KL(policy || ref_policy)

Reference policy is the checkpoint from the previous training phase (SFT).

Polyak averaging
─────────────────
Target network updated via exponential moving average:
    θ_target = polyak * θ_target + (1 - polyak) * θ_current
Prevents oscillation when the policy updates are large.

Run
────
    python -m training.grpo_trainer --epochs 30 --group-size 8

Or import and use GRPOTrainer directly for custom training loops.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from training.forensic_synthesis import (
    ContrastivePairGenerator,
    PrioritizedReplayBuffer,
    Transition,
    pairs_to_transitions,
)
from training.verifier import LogicalIntegrityVerifier, STEP_RISK_WEIGHTS

MODEL_DIR = Path("training/models")
DATA_PATH = Path("training/data/traces.jsonl")

ACTION_NAMES = {0: "ALLOW", 1: "QUARANTINE", 2: "ROLLBACK"}
N_ACTIONS = 3


# ─── Policy Network ───────────────────────────────────────────────────────────

def build_policy_net(input_dim: int) -> "torch.nn.Module":
    """
    TF-IDF features → softmax over {ALLOW, QUARANTINE, ROLLBACK}.
    Deeper than the old binary MLP: 3-layer with LayerNorm for stability.
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, N_ACTIONS),
        # No softmax here — use log_softmax in loss for numerical stability
    )


# ─── GRPO Config ─────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    group_size:   int   = 8       # G completions per trace
    clip_eps:     float = 0.20    # PPO-style clipping range
    kl_coef:      float = 0.04    # KL penalty weight
    lambda_v:     float = 0.35    # verifier reward weight
    polyak:       float = 0.995   # target network EMA
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    batch_size:   int   = 32
    epochs:       int   = 30
    sft_epochs:   int   = 5       # warm-up with step-aware SFT before GRPO
    buffer_size:  int   = 5_000
    seed:         int   = 42


# ─── Training History ─────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    epochs:          list[int]   = field(default_factory=list)
    policy_loss:     list[float] = field(default_factory=list)
    kl_div:          list[float] = field(default_factory=list)
    mean_reward:     list[float] = field(default_factory=list)
    mean_advantage:  list[float] = field(default_factory=list)
    verifier_reward: list[float] = field(default_factory=list)
    val_f1:          list[float] = field(default_factory=list)
    buffer_stats:    list[dict]  = field(default_factory=list)

    def record(self, **kwargs) -> None:
        for k, v in kwargs.items():
            getattr(self, k).append(round(float(v), 5) if isinstance(v, float) else v)

    def to_dict(self) -> dict:
        return {
            "epochs":          self.epochs,
            "policy_loss":     self.policy_loss,
            "kl_div":          self.kl_div,
            "mean_reward":     self.mean_reward,
            "mean_advantage":  self.mean_advantage,
            "verifier_reward": self.verifier_reward,
            "val_f1":          self.val_f1,
        }


# ─── GRPO Trainer ─────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    Full GRPO training loop for the TraceGuard policy.

    Training stages
    ───────────────
    Stage 1 — Step-Aware SFT (sft_epochs):
        Standard cross-entropy on the original trace dataset,
        with loss weighted by STEP_RISK_WEIGHTS[agent_step].
        This gives the policy a sensible starting point before RL.

    Stage 2 — GRPO (epochs):
        For each batch of traces from the prioritized replay buffer:
        a. Sample G actions from the current policy for each trace.
        b. Compute environment rewards via the reward fn.
        c. Compute verifier rewards via LogicalIntegrityVerifier.
        d. Combined reward = env_reward + lambda_v * verifier_reward.
        e. Group-relative advantage = (R_i - mean_group) / (std_group + eps).
        f. GRPO surrogate loss = -clipped(ratio * advantage).
        g. KL penalty = KL(policy || ref_policy).
        h. Update policy. Polyak-update target net.
    """

    def __init__(self, cfg: GRPOConfig = GRPOConfig()):
        self.cfg = cfg
        self.verifier = LogicalIntegrityVerifier(lambda_v=cfg.lambda_v)
        self.buffer = PrioritizedReplayBuffer(capacity=cfg.buffer_size)
        self.history = TrainingHistory()

        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_traces(self) -> tuple[list[str], list[int], list[str]]:
        """Load traces from JSONL. Returns (texts, labels, agent_steps)."""
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Run `python training/dataset.py` first. Missing: {DATA_PATH}")

        texts, labels, steps = [], [], []
        step_cycle = ["coder", "reviewer", "tester", "deployer"]
        for i, line in enumerate(DATA_PATH.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
            steps.append(step_cycle[i % 4])
        return texts, labels, steps

    # ── Vectorizer ────────────────────────────────────────────────────────────

    def _fit_vectorizer(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        vec.fit(texts)
        return vec

    # ── Step-Aware SFT loss ───────────────────────────────────────────────────

    def _sft_loss(self, logits, targets, agent_steps: list[str]):
        """
        Cross-entropy weighted by step risk.
        Deployer steps (risk=1.5) contribute 1.5× to the loss.
        """
        import torch
        import torch.nn.functional as F
        weights = torch.tensor(
            [STEP_RISK_WEIGHTS.get(s, 1.0) for s in agent_steps],
            dtype=torch.float32,
        )
        ce = F.cross_entropy(logits, targets, reduction="none")
        return (ce * weights).mean()

    # ── GRPO core ─────────────────────────────────────────────────────────────

    def _group_relative_advantage(self, rewards: list[float]) -> list[float]:
        """
        GRPO advantage: normalize rewards within the group.
        advantage_i = (R_i - mean_G) / (std_G + eps)
        """
        r = np.array(rewards, dtype=np.float64)
        mean_r = r.mean()
        std_r  = r.std() + 1e-8
        return list((r - mean_r) / std_r)

    def _compute_kl(self, log_probs_current, log_probs_ref) -> float:
        """KL(current || ref) = sum(exp(log_current) * (log_current - log_ref))"""
        import torch
        import torch.nn.functional as F
        p = torch.exp(log_probs_current)
        kl = (p * (log_probs_current - log_probs_ref)).sum(dim=-1).mean()
        return float(kl.item())

    def _grpo_surrogate(self, log_probs, log_probs_old, advantages, clip_eps: float):
        """
        Clipped GRPO surrogate (identical shape to PPO):
            ratio = exp(log_p_new - log_p_old)
            surrogate = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        """
        import torch
        ratios = torch.exp(log_probs - log_probs_old.detach())
        surr1  = ratios * advantages
        surr2  = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def _polyak_update(self, policy, target):
        """EMA update: θ_target = polyak * θ_target + (1-polyak) * θ_policy"""
        tau = 1.0 - self.cfg.polyak
        for p_params, t_params in zip(policy.parameters(), target.parameters()):
            t_params.data.copy_(tau * p_params.data + self.cfg.polyak * t_params.data)

    # ── Environment reward fn ─────────────────────────────────────────────────

    @staticmethod
    def _env_reward(action: int, is_attack: bool) -> float:
        if action == 0:   # ALLOW
            return 0.5 if not is_attack else -2.0
        else:             # QUARANTINE or ROLLBACK
            return 1.0 if is_attack else -1.0

    # ── Validation F1 ─────────────────────────────────────────────────────────

    def _val_f1(self, policy, vectorizer, X_val, y_val) -> float:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import f1_score
        policy.eval()
        with torch.no_grad():
            logits = policy(torch.FloatTensor(X_val))
            probs  = F.softmax(logits, dim=-1).numpy()
        # ALLOW=0 → label 0 (clean), QUARANTINE/ROLLBACK → label 1 (attack)
        preds = (np.argmax(probs, axis=1) > 0).astype(int)
        policy.train()
        return f1_score(y_val, preds, zero_division=0)

    # ── Main training entry ───────────────────────────────────────────────────

    def train(self) -> TrainingHistory:
        import torch
        import torch.nn.functional as F
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg

        # ── 1. Load data ───────────────────────────────────────────────────
        print("[GRPO] Loading data...")
        texts, labels, steps = self._load_traces()
        print(f"  {len(texts)} traces  |  attacks: {sum(labels)}")

        vectorizer = self._fit_vectorizer(texts)
        vec_path = MODEL_DIR / "vectorizer.pkl"
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)

        X = vectorizer.transform(texts).toarray().astype(np.float32)
        y = np.array(labels)

        X_tr, X_val, y_tr, y_val, steps_tr, _ = train_test_split(
            X, y, steps, test_size=0.2, random_state=cfg.seed, stratify=y
        )

        # ── 2. Build contrastive pairs → fill replay buffer ────────────────
        print("[GRPO] Synthesizing contrastive pairs...")
        clean_traces = [texts[i] for i in range(len(texts)) if labels[i] == 0]
        gen = ContrastivePairGenerator(seed=cfg.seed)
        pairs = gen.generate_batch(clean_traces, n_pairs=min(len(clean_traces) * 2, 1000))

        def _overseer_fn(trace, is_atk):
            # Minimal heuristic for buffer population (no model yet)
            from env.overseer import HeuristicScorer
            scorer = HeuristicScorer()
            score, _, _ = scorer.score(trace)
            action = 2 if score >= 0.85 else 1 if score >= 0.65 else 0
            reward = self._env_reward(action, is_atk)
            return score, action, reward

        transitions = pairs_to_transitions(pairs, _overseer_fn)
        self.buffer.push_many(transitions)
        print(f"  Buffer populated: {len(self.buffer)} transitions")
        print(f"  Buffer stats: {self.buffer.stats()}")

        # ── 3. Build policy and reference policy ──────────────────────────
        input_dim = X.shape[1]
        policy    = build_policy_net(input_dim)
        ref_policy = build_policy_net(input_dim)   # frozen reference
        target_net = build_policy_net(input_dim)   # Polyak-updated target

        optimizer = torch.optim.Adam(
            policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # ── 4. Stage 1: Step-Aware SFT warm-up ────────────────────────────
        print(f"\n[GRPO] Stage 1: Step-Aware SFT ({cfg.sft_epochs} epochs)...")
        X_tr_t = torch.FloatTensor(X_tr)

        # Convert binary labels to 3-class:
        # label=0 (clean) → action 0 (ALLOW)
        # label=1 (attack) → action 2 (ROLLBACK) — most conservative
        y_tr_3class = torch.LongTensor([0 if l == 0 else 2 for l in y_tr])

        for epoch in range(cfg.sft_epochs):
            policy.train()
            perm = torch.randperm(len(X_tr_t))
            total_loss = 0.0
            n_batches  = 0

            for i in range(0, len(X_tr_t), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                xb  = X_tr_t[idx]
                yb  = y_tr_3class[idx]
                sb  = [steps_tr[j] for j in idx.tolist()]

                optimizer.zero_grad()
                logits = policy(xb)
                loss = self._sft_loss(logits, yb, sb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)
            f1 = self._val_f1(policy, vectorizer, X_val, y_val)
            print(f"  SFT Epoch {epoch+1:02d}/{cfg.sft_epochs}  loss={avg_loss:.4f}  val_f1={f1:.4f}")

        # Copy SFT weights to ref_policy and target_net
        ref_policy.load_state_dict({k: v.clone() for k, v in policy.state_dict().items()})
        target_net.load_state_dict({k: v.clone() for k, v in policy.state_dict().items()})
        ref_policy.eval()  # frozen — never updated

        # ── 5. Stage 2: GRPO ──────────────────────────────────────────────
        print(f"\n[GRPO] Stage 2: GRPO ({cfg.epochs} epochs, G={cfg.group_size})...")

        for epoch in range(cfg.epochs):
            self.buffer.anneal_beta(epoch, cfg.epochs)
            policy.train()

            # Sample batch from prioritized replay buffer
            batch_transitions, is_weights = self.buffer.sample(cfg.batch_size * cfg.group_size)
            if not batch_transitions:
                print(f"  Epoch {epoch+1}: buffer empty, skipping.")
                continue

            is_w_t = torch.FloatTensor(is_weights)

            # Group transitions: G per "trace group"
            # Each group = G different actions sampled for the same or similar traces
            group_rewards, group_log_probs, group_log_probs_old = [], [], []
            group_advantages_all = []
            verifier_rewards_all = []
            kl_divs = []

            # Vectorize batch traces
            batch_texts = [t.trace for t in batch_transitions]
            X_batch = vectorizer.transform(batch_texts).toarray().astype(np.float32)
            X_batch_t = torch.FloatTensor(X_batch)

            with torch.no_grad():
                ref_logits = ref_policy(X_batch_t)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # Current policy forward
            logits = policy(X_batch_t)
            log_probs = F.log_softmax(logits, dim=-1)

            # For each item in batch, simulate G completions
            # (In practice with a small MLP, we use Gumbel sampling for diversity)
            all_grpo_loss = torch.tensor(0.0, requires_grad=True)
            total_kl = 0.0
            total_reward = 0.0
            total_v_reward = 0.0

            # Build group batches
            n_groups = len(batch_transitions) // cfg.group_size
            if n_groups == 0:
                n_groups = 1

            batch_policy_loss = []

            for g_start in range(0, len(batch_transitions) - cfg.group_size + 1, cfg.group_size):
                group = batch_transitions[g_start: g_start + cfg.group_size]
                g_X = X_batch_t[g_start: g_start + cfg.group_size]
                g_lp_ref = ref_log_probs[g_start: g_start + cfg.group_size]

                g_logits = policy(g_X)
                g_log_probs = F.log_softmax(g_logits, dim=-1)

                # Sample actions (Gumbel-softmax for differentiable sampling)
                g_actions_soft = F.gumbel_softmax(g_logits, tau=1.0, hard=True)
                g_actions = g_actions_soft.argmax(dim=-1)

                # Compute combined rewards for each group member
                g_rewards = []
                g_v_rewards = []
                for i, trans in enumerate(group):
                    action_int = int(g_actions[i].item())
                    env_r = self._env_reward(action_int, trans.is_attack)
                    v_result = self.verifier.verify(trans.trace, trans.agent_step)
                    v_r = v_result.verifier_reward
                    combined_r = env_r + cfg.lambda_v * v_r
                    g_rewards.append(combined_r)
                    g_v_rewards.append(v_r)

                # Group-relative advantages
                advantages = self._group_relative_advantage(g_rewards)
                adv_t = torch.FloatTensor(advantages)

                # Log probs for taken actions
                taken_lp = g_log_probs[torch.arange(len(group)), g_actions]
                taken_lp_ref = g_lp_ref[torch.arange(len(group)), g_actions]

                # GRPO surrogate
                surr_loss = self._grpo_surrogate(taken_lp, taken_lp_ref, adv_t, cfg.clip_eps)

                # KL penalty
                kl = self._compute_kl(g_log_probs, g_lp_ref)
                group_loss = surr_loss + cfg.kl_coef * kl

                batch_policy_loss.append(group_loss)
                total_reward  += np.mean(g_rewards)
                total_v_reward += np.mean(g_v_rewards)
                total_kl += kl

            if not batch_policy_loss:
                continue

            final_loss = torch.stack(batch_policy_loss).mean()
            optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # Polyak update target network
            self._polyak_update(policy, target_net)

            n_g = max(n_groups, 1)
            avg_reward  = total_reward  / n_g
            avg_v_rwd   = total_v_reward / n_g
            avg_kl      = total_kl      / n_g
            f1 = self._val_f1(policy, vectorizer, X_val, y_val)

            print(
                f"  GRPO Epoch {epoch+1:02d}/{cfg.epochs}  "
                f"loss={final_loss.item():.4f}  "
                f"reward={avg_reward:.3f}  "
                f"v_reward={avg_v_rwd:.3f}  "
                f"kl={avg_kl:.4f}  "
                f"val_f1={f1:.4f}"
            )

            self.history.record(
                epochs=epoch + cfg.sft_epochs + 1,
                policy_loss=final_loss.item(),
                kl_div=avg_kl,
                mean_reward=avg_reward,
                mean_advantage=0.0,   # averaged out in GRPO by definition
                verifier_reward=avg_v_rwd,
                val_f1=f1,
            )

        # ── 6. Save ───────────────────────────────────────────────────────
        policy_path = MODEL_DIR / "grpo_policy.pt"
        target_path = MODEL_DIR / "grpo_target.pt"
        import torch
        torch.save(policy.state_dict(), policy_path)
        torch.save(target_net.state_dict(), target_path)
        print(f"\n[GRPO] Policy saved → {policy_path}")

        hist_path = MODEL_DIR / "grpo_history.json"
        hist_path.write_text(json.dumps(self.history.to_dict(), indent=2))
        print(f"[GRPO] History saved → {hist_path}")

        # Save metrics
        final_f1 = self.history.val_f1[-1] if self.history.val_f1 else 0.0
        metrics = {
            "model":           "GRPO_PolicyNet",
            "grpo_epochs":     cfg.epochs,
            "sft_epochs":      cfg.sft_epochs,
            "group_size":      cfg.group_size,
            "clip_eps":        cfg.clip_eps,
            "kl_coef":         cfg.kl_coef,
            "lambda_v":        cfg.lambda_v,
            "polyak":          cfg.polyak,
            "final_val_f1":    round(final_f1, 4),
            "buffer_stats":    self.buffer.stats(),
        }
        (MODEL_DIR / "grpo_metrics.json").write_text(json.dumps(metrics, indent=2))

        return self.history


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TraceGuard GRPO Trainer")
    parser.add_argument("--epochs",      type=int,   default=30,    help="GRPO epochs")
    parser.add_argument("--sft-epochs",  type=int,   default=5,     help="SFT warm-up epochs")
    parser.add_argument("--group-size",  type=int,   default=8,     help="G completions per trace")
    parser.add_argument("--lr",          type=float, default=3e-4,  help="Learning rate")
    parser.add_argument("--kl-coef",     type=float, default=0.04,  help="KL penalty coefficient")
    parser.add_argument("--lambda-v",    type=float, default=0.35,  help="Verifier reward weight")
    parser.add_argument("--clip-eps",    type=float, default=0.20,  help="PPO clip epsilon")
    parser.add_argument("--seed",        type=int,   default=42,    help="Random seed")
    args = parser.parse_args()

    cfg = GRPOConfig(
        epochs=args.epochs,
        sft_epochs=args.sft_epochs,
        group_size=args.group_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        lambda_v=args.lambda_v,
        clip_eps=args.clip_eps,
        seed=args.seed,
    )

    trainer = GRPOTrainer(cfg)
    history = trainer.train()

    print("\n[GRPO] Training complete.")
    if history.val_f1:
        print(f"  Final val F1:      {history.val_f1[-1]:.4f}")
        print(f"  Final mean reward: {history.mean_reward[-1]:.4f}")
        print(f"  Final KL div:      {history.kl_div[-1]:.5f}")


if __name__ == "__main__":
    main()
