from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.schemas import BallDetection2D


class WasbSbdtDetector:
    """WASB-SBDT detector backend adapted to the local BallDetection2D interface."""

    def __init__(self, config: dict):
        self._cv2 = _import_required("cv2", "opencv-python")
        self._torch = _import_required("torch", "torch")
        omega_conf = _import_required("omegaconf", "omegaconf")

        bt_config = config.get("ball_tracking", {})
        wasb_config = bt_config.get("wasb_sbdt", {})

        self._repo_src = _resolve_repo_src(wasb_config)
        self._model_path = _resolve_model_path(config, self._repo_src)
        self._device = self._resolve_device(wasb_config)

        model_cfg = _load_yaml(self._repo_src / "configs" / "model" / "wasb.yaml")
        dataloader_cfg = _load_yaml(self._repo_src / "configs" / "dataloader" / "default.yaml")

        self._frames_in = int(model_cfg["frames_in"])
        self._frames_out = int(model_cfg["frames_out"])
        self._input_wh = (int(model_cfg["inp_width"]), int(model_cfg["inp_height"]))
        self._out_scale = int(model_cfg["out_scales"][0])

        self._score_threshold = float(wasb_config.get("score_threshold", 0.5))
        self._blob_det_method = str(wasb_config.get("blob_det_method", "concomp"))
        self._use_hm_weight = bool(wasb_config.get("use_hm_weight", True))
        self._hm_sigma = float(dataloader_cfg["heatmap"]["sigmas"][self._out_scale])

        self._recent_frames: deque[np.ndarray] = deque(maxlen=self._frames_in)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        cfg = omega_conf.OmegaConf.create({"model": model_cfg})
        self._model = self._build_model(cfg)

    def detect_frame(
        self, frame: np.ndarray, prev_frames: list[np.ndarray] | None = None
    ) -> BallDetection2D | None:
        if prev_frames is None:
            self._recent_frames.append(frame)
            if len(self._recent_frames) < self._frames_in:
                return None
            frames = list(self._recent_frames)
        else:
            frames = [*prev_frames, frame][-self._frames_in :]
            if len(frames) < self._frames_in:
                return None

        input_tensor, output_transform_inv = self._prepare_clip(frames)

        with self._torch.no_grad():
            preds = self._model(input_tensor)
            heatmaps = preds[self._out_scale].sigmoid().detach().cpu().numpy()

        output_index = min(self._frames_out - 1, heatmaps.shape[1] - 1)
        candidate = self._best_candidate(heatmaps[0, output_index])
        if candidate is None:
            return None

        image_xy = _affine_transform(np.array([candidate["x"], candidate["y"]]), output_transform_inv)
        height, width = frame.shape[:2]
        if not (0.0 <= image_xy[0] < width and 0.0 <= image_xy[1] < height):
            return None

        return BallDetection2D(
            frame=0,
            time_s=0.0,
            image_xy=(float(image_xy[0]), float(image_xy[1])),
            confidence=float(candidate["confidence"]),
            visibility="visible",
            source="wasb_sbdt",
        )

    def _resolve_device(self, wasb_config: dict) -> Any:
        requested = str(wasb_config.get("device", "auto"))
        if requested == "auto":
            requested = "cuda" if self._torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("WASB-SBDT requested CUDA but torch.cuda.is_available() is false")
        return self._torch.device(requested)

    def _build_model(self, cfg: Any) -> Any:
        if str(self._repo_src) not in sys.path:
            sys.path.insert(0, str(self._repo_src))

        try:
            from models import build_model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"Could not import WASB-SBDT model code from {self._repo_src}") from exc

        model = build_model(cfg)
        checkpoint = self._torch.load(str(self._model_path), map_location=self._device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(self._device)
        model.eval()
        return model

    def _prepare_clip(self, frames: list[np.ndarray]) -> tuple[Any, np.ndarray]:
        first_rgb = self._cv2.cvtColor(frames[0], self._cv2.COLOR_BGR2RGB)
        input_transform = _frame_transform(first_rgb.shape, self._input_wh, self._cv2)
        output_transform_inv = _frame_transform(first_rgb.shape, self._input_wh, self._cv2, inv=True)

        tensors = []
        for frame in frames:
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            warped = self._cv2.warpAffine(rgb, input_transform, self._input_wh, flags=self._cv2.INTER_LINEAR)
            normalized = (warped.astype(np.float32) / 255.0 - self._mean) / self._std
            tensors.append(normalized.transpose(2, 0, 1))

        clip = np.concatenate(tensors, axis=0)
        return self._torch.from_numpy(clip).unsqueeze(0).to(self._device), output_transform_inv

    def _best_candidate(self, heatmap: np.ndarray) -> dict[str, float] | None:
        if self._blob_det_method == "nms":
            candidates = self._nms_candidates(heatmap)
        else:
            candidates = self._connected_component_candidates(heatmap)

        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate["rank_score"])

    def _connected_component_candidates(self, heatmap: np.ndarray) -> list[dict[str, float]]:
        if float(np.max(heatmap)) <= self._score_threshold:
            return []

        _, thresholded = self._cv2.threshold(heatmap, self._score_threshold, 1, self._cv2.THRESH_BINARY)
        n_labels, labels = self._cv2.connectedComponents(thresholded.astype(np.uint8))

        candidates: list[dict[str, float]] = []
        for label in range(1, n_labels):
            ys, xs = np.where(labels == label)
            weights = heatmap[ys, xs]
            if weights.size == 0:
                continue

            if self._use_hm_weight:
                weight_sum = float(np.sum(weights))
                if weight_sum <= 0.0:
                    continue
                x = float(np.sum(xs * weights) / weight_sum)
                y = float(np.sum(ys * weights) / weight_sum)
                rank_score = weight_sum
            else:
                x = float(np.mean(xs))
                y = float(np.mean(ys))
                rank_score = float(weights.size)

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "rank_score": rank_score,
                    "confidence": _clamp01(float(np.max(weights))),
                }
            )

        return candidates

    def _nms_candidates(self, heatmap: np.ndarray) -> list[dict[str, float]]:
        heatmap_work = heatmap.copy()
        heatmap_original = heatmap.copy()
        height, width = heatmap.shape
        map_x, map_y = np.meshgrid(np.linspace(1, width, width), np.linspace(1, height, height))

        candidates: list[dict[str, float]] = []
        while True:
            cy, cx = np.unravel_index(np.argmax(heatmap_work), heatmap_work.shape)
            peak = float(heatmap_work[cy, cx])
            if peak <= self._score_threshold:
                break

            dist_map = (map_y - (cy + 1)) ** 2 + (map_x - (cx + 1)) ** 2
            region = dist_map <= self._hm_sigma**2
            ys, xs = np.where(region)
            weights = heatmap_original[region]
            if self._use_hm_weight:
                weight_sum = float(np.sum(weights))
                x = float(np.sum(xs * weights) / weight_sum)
                y = float(np.sum(ys * weights) / weight_sum)
                rank_score = weight_sum
            else:
                x = float(np.mean(xs))
                y = float(np.mean(ys))
                rank_score = float(weights.size)

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "rank_score": rank_score,
                    "confidence": _clamp01(peak),
                }
            )
            heatmap_work[region] = 0.0

        return candidates


def _resolve_repo_src(wasb_config: dict) -> Path:
    candidates = []
    configured = wasb_config.get("repo_path") or os.environ.get("WASB_SBDT_REPO")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path("third_party/WASB-SBDT"),
            Path("external/WASB-SBDT"),
            Path("../WASB-SBDT"),
        ]
    )

    for candidate in candidates:
        src = candidate / "src"
        if (src / "configs" / "model" / "wasb.yaml").exists():
            return src.resolve()
        if (candidate / "configs" / "model" / "wasb.yaml").exists():
            return candidate.resolve()

    raise RuntimeError("WASB-SBDT repo not found; set ball_tracking.wasb_sbdt.repo_path or WASB_SBDT_REPO")


def _resolve_model_path(config: dict, repo_src: Path) -> Path:
    bt_config = config.get("ball_tracking", {})
    wasb_config = bt_config.get("wasb_sbdt", {})
    dataset = str(wasb_config.get("dataset", "tennis"))
    configured = wasb_config.get("model_path") or bt_config.get("model_path")

    candidates = []
    if configured:
        path = Path(configured).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([Path.cwd() / path, repo_src / path, repo_src.parent / path])
    else:
        filename = f"wasb_{dataset}_best.pth.tar"
        models_dir = Path(config.get("models", {}).get("cache_dir", "data/models"))
        candidates.extend([models_dir / filename, repo_src.parent / "pretrained_weights" / filename])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    display = configured if configured else f"wasb_{dataset}_best.pth.tar"
    raise RuntimeError(f"WASB-SBDT model weights not found: {display}")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _import_required(module_name: str, package_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"WASB-SBDT backend requires {package_name}") from exc


def _frame_transform(
    shape: tuple[int, ...],
    input_wh: tuple[int, int],
    cv2_module: Any,
    inv: bool = False,
) -> np.ndarray:
    height, width = shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = np.array([max(height, width), max(height, width)], dtype=np.float32)
    src_w = scale[0]
    dst_w, dst_h = input_wh

    src_dir = np.array([0.0, src_w * -0.5], dtype=np.float32)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    src[2, :] = _third_point(src[0, :], src[1, :])
    dst[2, :] = _third_point(dst[0, :], dst[1, :])

    if inv:
        return cv2_module.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2_module.getAffineTransform(np.float32(src), np.float32(dst))


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _affine_transform(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
    return np.dot(transform, homogeneous)[:2]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
