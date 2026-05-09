"""Extract frames from a video file."""
from __future__ import annotations

from pathlib import Path

import click
from loguru import logger


@click.command()
@click.option("--video", required=True, type=click.Path(exists=True), help="Input video path")
@click.option("--out", default="data/frames", type=click.Path(), help="Output frames directory")
@click.option("--interval", default=1.0, type=float, help="Seconds between extracted frames")
@click.option("--max-frames", default=None, type=int, help="Maximum number of frames to extract")
def main(video: str, out: str, interval: float, max_frames: int | None) -> None:
    """Extract frames from video at specified interval."""
    logger.info(f"Extracting frames from {video}")
    logger.info(f"Interval: {interval}s, Output: {out}")

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    # TODO: Implement frame extraction with cv2.VideoCapture
    logger.warning("Frame extraction not yet implemented")


if __name__ == "__main__":
    main()
