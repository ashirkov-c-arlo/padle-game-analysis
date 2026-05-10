from __future__ import annotations

import json

import cv2
import numpy as np
from loguru import logger

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

_PROMPT = (
    "Find the scoreboard overlay in this broadcast frame. "
    "Detect the whole bounding box and read the current set and game score.\n"
    "Return bbox as [x1, y1, x2, y2] in RELATIVE coordinates (0-1000 range). "
    "Return ONLY valid JSON:\n"
    '{"bbox": [x1, y1, x2, y2], "players":[[player1_name, player2_name],'
    '[player3_name, player4_name]], "sets": [[t1, t2], ...], '
    '"game_score": ["pts1", "pts2"], "confidence": 0.0-1.0}\n'
    "If no scoreboard visible: "
    '{"bbox": null, "sets": null, "game_score": null, "confidence": 0.0}'
)

_bedrock_client = None


def is_vlm_available(config: dict) -> bool:
    return BOTO3_AVAILABLE and config.get("enabled", False)


def detect_scoreboard_vlm(
    frame: np.ndarray,
    config: dict,
) -> dict | None:
    """
    Send full frame to Bedrock VLM, get scoreboard bbox + score text.

    Returns dict with keys roi_bbox_xyxy, raw_text, sets, game_score, confidence.
    Returns None on any failure.
    """
    try:
        client = _get_bedrock_client(config)
        max_side = config.get("image_max_side", 1280)
        resized = _resize_frame(frame, max_side)
        jpeg_bytes = _encode_frame_jpeg(resized, quality=config.get("jpeg_quality", 80))

        response = client.converse(
            modelId=config.get("model_id", "qwen.qwen3-vl-235b-a22b"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": jpeg_bytes}}},
                        {"text": _PROMPT},
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.0,
            },
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        logger.debug("Raw VLM response text: {}", response_text[:200])
        return _parse_vlm_response(response_text, frame.shape[:2])
    except ClientError as e:
        logger.warning("Bedrock VLM call failed: {}", e)
        return None
    except (KeyError, IndexError, TypeError) as e:
        logger.warning("VLM response structure unexpected: {}", e)
        return None
    except Exception as e:
        logger.warning("VLM detection error: {}", e)
        return None


def _get_bedrock_client(config: dict):
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client

    _bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=config.get("region", "us-east-1"),
        config=BotoConfig(
            read_timeout=config.get("timeout_s", 30),
            retries={"max_attempts": 1},
        ),
    )
    model_id = config.get("model_id", "qwen.qwen3-vl-235b-a22b")
    logger.info("VLM model loaded: {}", model_id)
    return _bedrock_client


def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _parse_vlm_response(
    text: str,
    original_hw: tuple[int, int],
) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("VLM response is not valid JSON: {}", text[:200])
        return None

    bbox = data.get("bbox")
    roi = None
    if bbox and len(bbox) == 4:
        try:
            h, w = original_hw
            x1 = int(round(float(bbox[0]) / 1000.0 * w))
            y1 = int(round(float(bbox[1]) / 1000.0 * h))
            x2 = int(round(float(bbox[2]) / 1000.0 * w))
            y2 = int(round(float(bbox[3]) / 1000.0 * h))
            coords = [
                max(0, min(x1, w)),
                max(0, min(y1, h)),
                max(0, min(x2, w)),
                max(0, min(y2, h)),
            ]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                roi = tuple(coords)
        except (ValueError, TypeError):
            pass

    sets_raw = data.get("sets")
    sets_parsed = None
    if sets_raw and isinstance(sets_raw, list):
        try:
            sets_parsed = [(int(s[0]), int(s[1])) for s in sets_raw if len(s) == 2]
            if not sets_parsed:
                sets_parsed = None
        except (ValueError, TypeError, IndexError):
            sets_parsed = None

    game_score = data.get("game_score")
    game_score_parsed = None
    if game_score and isinstance(game_score, list) and len(game_score) == 2:
        game_score_parsed = (str(game_score[0]), str(game_score[1]))

    raw_parts = []
    if sets_parsed:
        raw_parts.append(" ".join(f"{a}-{b}" for a, b in sets_parsed))
    if game_score_parsed:
        raw_parts.append(f"{game_score_parsed[0]}-{game_score_parsed[1]}")
    raw_text = " ".join(raw_parts) if raw_parts else None

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(confidence, 1.0))

    if roi is None and sets_parsed is None and game_score_parsed is None:
        return None

    return {
        "roi_bbox_xyxy": roi,
        "raw_text": raw_text,
        "sets": sets_parsed,
        "game_score": game_score_parsed,
        "confidence": confidence,
    }
