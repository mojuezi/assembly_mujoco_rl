"""Gated multimodal feature extractors for peg-insertion SAC policies.

The modules in this file are intentionally SB3-friendly: auxiliary heads are
computed and cached for inspection, but the feature extractor still returns a
single fused tensor that can be used unchanged with SAC + MultiInputPolicy.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


CONTACT_STATE_DIM = 5  # no_contact/contact/alignment/insertion/jam
POSE_PRIOR_DIM = 6  # dx, dy, dz, droll, dpitch, dyaw
DEFAULT_PHASE_DIM = 5


def _flat_dim(space: spaces.Space) -> int:
    """Return the flattened dimension for a Gym space."""
    return int(np.prod(space.shape))


def _batch_size_from_observations(observations: Dict[str, th.Tensor]) -> int:
    """Infer batch size from the first tensor in a SB3 observation dict."""
    if not observations:
        raise ValueError("observations must contain at least one tensor")
    first = next(iter(observations.values()))
    return int(first.shape[0])


def _zeros_like_batch(
    observations: Dict[str, th.Tensor],
    dim: int,
) -> th.Tensor:
    """Create a zero tensor on the same device/dtype as the observation batch."""
    first = next(iter(observations.values()))
    return first.new_zeros((_batch_size_from_observations(observations), dim))


def _phase_tensor(
    observations: Dict[str, th.Tensor],
    phase_dim: int,
) -> th.Tensor:
    """Return flattened phase features or zeros when the obs mode has no phase."""
    if "phase" not in observations:
        return _zeros_like_batch(observations, phase_dim)
    phase = th.flatten(observations["phase"], start_dim=1)
    if phase.shape[1] == phase_dim:
        return phase

    # Be tolerant of future wrappers with a different phase length: pad/truncate
    # so the gate MLP has the static input size it was built with.
    if phase.shape[1] > phase_dim:
        return phase[:, :phase_dim]
    pad = phase.new_zeros((phase.shape[0], phase_dim - phase.shape[1]))
    return th.cat([phase, pad], dim=1)


def _mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    """Build a compact ReLU MLP."""
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class VisualPoseEncoder(nn.Module):
    """Encode optional RGB/depth observations and predict visual auxiliary heads.

    Inputs are read from an observation dict. Each available visual stream is a
    CHW tensor with batch shape ``(B, C, H, W)``. The returned ``z_visual`` has
    shape ``(B, latent_dim)``. ``pose_prior_head`` predicts a 6-D relative pose
    prior; ``visual_confidence_head`` predicts a scalar confidence in ``[0, 1]``.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        latent_dim: int = 128,
        stream_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.visual_keys = [key for key in ("rgb", "depth") if key in observation_space.spaces]
        self.stream_cnns = nn.ModuleDict()

        for key in self.visual_keys:
            channels = int(observation_space[key].shape[0])
            self.stream_cnns[key] = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, stream_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        fusion_input_dim = max(1, len(self.visual_keys)) * stream_dim
        self.fusion = _mlp(fusion_input_dim, (128,), self.latent_dim)
        self.pose_prior_head = nn.Linear(self.latent_dim, POSE_PRIOR_DIM)
        self.visual_confidence_head = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())

    @property
    def has_visual(self) -> bool:
        return len(self.visual_keys) > 0

    def forward(self, observations: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if not self.has_visual:
            z_visual = _zeros_like_batch(observations, self.latent_dim)
        else:
            stream_features = [self.stream_cnns[key](observations[key]) for key in self.visual_keys]
            z_visual = self.fusion(th.cat(stream_features, dim=1))

        pose_prior = self.pose_prior_head(z_visual)
        visual_confidence = self.visual_confidence_head(z_visual)
        return z_visual, pose_prior, visual_confidence


class ForceHistoryEncoder(nn.Module):
    """Encode optional wrench history plus phase into a force/contact latent.

    ``force`` is expected as ``(B, history_len, 6)``. The encoder also accepts a
    flattened fallback, reshaping it to history records when possible. The output
    ``contact_state_logits`` has five classes: no_contact, contact, alignment,
    insertion, jam.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        latent_dim: int = 128,
        phase_dim: int = DEFAULT_PHASE_DIM,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.phase_dim = int(phase_dim)
        self.has_force = "force" in observation_space.spaces

        if self.has_force:
            force_shape = observation_space["force"].shape
            self.history_len = int(force_shape[0]) if len(force_shape) >= 2 else max(1, _flat_dim(observation_space["force"]) // 6)
            wrench_dim = int(force_shape[-1]) if len(force_shape) >= 2 else 6
            if wrench_dim != 6:
                raise ValueError(f"force observation must end with 6 wrench values, got shape={force_shape}")
            self.force_cnn = nn.Sequential(
                nn.Conv1d(6, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            input_dim = 64 + self.phase_dim
        else:
            self.history_len = 0
            self.force_cnn = None
            input_dim = self.phase_dim

        self.fusion = _mlp(input_dim, (128,), self.latent_dim)
        self.contact_state_logits = nn.Linear(self.latent_dim, CONTACT_STATE_DIM)

    def _encode_force(self, force: th.Tensor) -> th.Tensor:
        if force.ndim == 2:
            force = force.view(force.shape[0], self.history_len, 6)
        # Conv1d consumes channels-first wrench history: (B, 6, history_len).
        force = force.transpose(1, 2)
        return self.force_cnn(force)

    def forward(self, observations: Dict[str, th.Tensor], phase: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if self.has_force:
            force_features = self._encode_force(observations["force"])
            inputs = th.cat([force_features, phase], dim=1)
        else:
            inputs = phase

        z_force = self.fusion(inputs)
        contact_logits = self.contact_state_logits(z_force)
        return z_force, contact_logits


class ProprioEncoder(nn.Module):
    """Encode proprioceptive state with a small MLP."""

    def __init__(self, observation_space: spaces.Dict, latent_dim: int = 128):
        super().__init__()
        if "proprio" not in observation_space.spaces:
            raise ValueError("GatedFusionExtractor requires a 'proprio' observation")
        self.latent_dim = int(latent_dim)
        proprio_dim = _flat_dim(observation_space["proprio"])
        self.mlp = _mlp(proprio_dim, (128, 128), self.latent_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        proprio = th.flatten(observations["proprio"], start_dim=1)
        return self.mlp(proprio)


class PredictionEncoder(nn.Module):
    """Fuse force/proprio/phase into a predictive latent with reserved heads."""

    def __init__(
        self,
        force_dim: int = 128,
        proprio_dim: int = 128,
        phase_dim: int = DEFAULT_PHASE_DIM,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        input_dim = int(force_dim) + int(proprio_dim) + int(phase_dim)
        self.mlp = _mlp(input_dim, (128,), self.latent_dim)

        # Reserved for future auxiliary objectives; currently cached only.
        self.next_wrench_head = nn.Linear(self.latent_dim, 6)
        self.next_contact_head = nn.Linear(self.latent_dim, CONTACT_STATE_DIM)
        self.next_progress_head = nn.Linear(self.latent_dim, 1)

    def forward(self, z_force: th.Tensor, z_proprio: th.Tensor, phase: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        z_pred = self.mlp(th.cat([z_force, z_proprio, phase], dim=1))
        next_wrench = self.next_wrench_head(z_pred)
        next_contact = self.next_contact_head(z_pred)
        next_progress = self.next_progress_head(z_pred)
        return z_pred, next_wrench, next_contact, next_progress


class GatedFusionExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor with phase-aware gated fusion over four experts.

    The four experts are visual, force, proprio, and predictive latents. Each is
    projected to ``expert_dim`` and mixed by a softmax gate. ``forward`` returns
    only the final ``features_dim`` tensor expected by SAC + MultiInputPolicy;
    auxiliary predictions and gate values are available in ``self.last_aux`` for
    debugging or future loss wiring.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        latent_dim: int = 128,
        visual_latent_dim: int | None = None,
        force_latent_dim: int | None = None,
        proprio_latent_dim: int | None = None,
        pred_latent_dim: int | None = None,
        expert_dim: int = 128,
        gate_temperature: float = 1.0,
        enable_predictive_encoder: bool = True,
        use_phase_gate_prior: bool = True,
        phase_prior_strength: float = 1.0,
    ):
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("GatedFusionExtractor expects a gymnasium.spaces.Dict observation space")

        self.output_features_dim = int(features_dim)
        self.latent_dim = int(latent_dim)
        self.visual_latent_dim = int(visual_latent_dim or latent_dim)
        self.force_latent_dim = int(force_latent_dim or latent_dim)
        self.proprio_latent_dim = int(proprio_latent_dim or latent_dim)
        self.pred_latent_dim = int(pred_latent_dim or latent_dim)
        self.expert_dim = int(expert_dim)
        self.gate_temperature = max(float(gate_temperature), 1.0e-6)
        self.enable_predictive_encoder = bool(enable_predictive_encoder)
        self.use_phase_gate_prior = bool(use_phase_gate_prior)
        self.phase_dim = _flat_dim(observation_space["phase"]) if "phase" in observation_space.spaces else DEFAULT_PHASE_DIM
        self.phase_prior_strength = float(phase_prior_strength)

        self.visual_encoder = VisualPoseEncoder(observation_space, latent_dim=self.visual_latent_dim)
        self.force_encoder = ForceHistoryEncoder(observation_space, latent_dim=self.force_latent_dim, phase_dim=self.phase_dim)
        self.proprio_encoder = ProprioEncoder(observation_space, latent_dim=self.proprio_latent_dim)
        self.prediction_encoder = PredictionEncoder(
            force_dim=self.force_latent_dim,
            proprio_dim=self.proprio_latent_dim,
            phase_dim=self.phase_dim,
            latent_dim=self.pred_latent_dim,
        )

        self.expert_projections = nn.ModuleDict(
            {
                "visual": nn.Linear(self.visual_latent_dim, self.expert_dim),
                "force": nn.Linear(self.force_latent_dim, self.expert_dim),
                "proprio": nn.Linear(self.proprio_latent_dim, self.expert_dim),
                "pred": nn.Linear(self.pred_latent_dim, self.expert_dim),
            }
        )
        self.register_buffer(
            "expert_available",
            th.tensor(
                [
                    self.visual_encoder.has_visual,
                    self.force_encoder.has_force,
                    True,  # proprio is required
                    self.enable_predictive_encoder,
                ],
                dtype=th.bool,
            ),
            persistent=False,
        )

        gate_input_dim = (
            self.visual_latent_dim
            + self.force_latent_dim
            + self.proprio_latent_dim
            + self.pred_latent_dim
            + self.phase_dim
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(self.expert_dim, self.output_features_dim),
            nn.ReLU(),
        )
        self.last_aux: Dict[str, th.Tensor] = {}

    def _phase_prior_logits(self, phase: th.Tensor) -> th.Tensor:
        """Build additive gate logits from phase heuristics.

        Phase convention from wrappers.py: [contact_flag, jam_flag, force_norm,
        torque_norm, insertion_progress]. With no phase observation the tensor is
        zero, so the prior gently favors vision before contact.
        """
        contact = phase[:, 0].clamp(0.0, 1.0) if phase.shape[1] >= 1 else phase.new_zeros(phase.shape[0])
        jam = phase[:, 1].clamp(0.0, 1.0) if phase.shape[1] >= 2 else phase.new_zeros(phase.shape[0])
        no_contact = 1.0 - contact

        prior = phase.new_zeros((phase.shape[0], 4))
        prior[:, 0] += 1.25 * no_contact  # visual expert: approach / pre-contact
        prior[:, 1] += 1.25 * contact + 1.00 * jam  # force expert: contact and jam
        prior[:, 2] += 0.25  # proprio remains a stable low-level anchor
        prior[:, 3] += 1.00 * jam  # prediction expert: jam recovery / temporal reasoning
        return prior * self.phase_prior_strength

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        phase = _phase_tensor(observations, self.phase_dim)

        z_visual, pose_prior, visual_confidence = self.visual_encoder(observations)
        z_force, contact_logits = self.force_encoder(observations, phase)
        z_proprio = self.proprio_encoder(observations)
        if self.enable_predictive_encoder:
            z_pred, next_wrench, next_contact, next_progress = self.prediction_encoder(z_force, z_proprio, phase)
        else:
            z_pred = z_proprio.new_zeros((z_proprio.shape[0], self.pred_latent_dim))
            next_wrench = z_proprio.new_zeros((z_proprio.shape[0], 6))
            next_contact = z_proprio.new_zeros((z_proprio.shape[0], CONTACT_STATE_DIM))
            next_progress = z_proprio.new_zeros((z_proprio.shape[0], 1))

        expert_latents = {
            "visual": z_visual,
            "force": z_force,
            "proprio": z_proprio,
            "pred": z_pred,
        }
        experts = th.stack(
            [self.expert_projections[name](expert_latents[name]) for name in ("visual", "force", "proprio", "pred")],
            dim=1,
        )  # (B, 4, expert_dim)

        gate_inputs = th.cat([z_visual, z_force, z_proprio, z_pred, phase], dim=1)
        gate_logits = self.gate_mlp(gate_inputs)
        if self.use_phase_gate_prior:
            gate_logits = gate_logits + self._phase_prior_logits(phase)
        # Keep four gate slots for logging/compatibility, but do not let missing
        # modalities contribute through projection biases.
        gate_logits = gate_logits.masked_fill(~self.expert_available.unsqueeze(0), -1.0e9)
        gates = th.softmax(gate_logits / self.gate_temperature, dim=1)  # (B, 4), ordered visual/force/proprio/pred

        fused_expert = th.sum(experts * gates.unsqueeze(-1), dim=1)
        features = self.output_mlp(fused_expert)

        # Cache detached tensors to avoid accidentally retaining SAC graphs while
        # still making the auxiliary signals easy to inspect during debugging.
        self.last_aux = {
            "gate": gates.detach(),
            "gate_logits": gate_logits.detach(),
            "pose_prior": pose_prior.detach(),
            "visual_confidence": visual_confidence.detach(),
            "contact_state_logits": contact_logits.detach(),
            "next_wrench": next_wrench.detach(),
            "next_contact": next_contact.detach(),
            "next_progress": next_progress.detach(),
        }
        return features
