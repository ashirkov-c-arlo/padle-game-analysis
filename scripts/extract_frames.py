"""Extract frames from a video file."""
from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from src.logging_config import LOG_LEVELS, configure_logging


@click.command()
@click.option("--video", required=True, type=click.Path(exists=True), help="Input video path")
@click.option("--out", default="data/frames", type=click.Path(), help="Output frames directory")
@click.option("--interval", default=1.0, type=float, help="Seconds between extracted frames")
@click.option("--max-frames", default=None, type=int, help="Maximum number of frames to extract")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log level. Defaults to PADEL_CV_LOG_LEVEL or INFO.",
)
def main(video: str, out: str, interval: float, max_frames: int | None, log_level: str | None) -> None:
    """Extract frames from video at specified interval."""
    configure_logging(log_level)
    logger.info("Extracting frames: video={}, output={}", video, out)
    logger.debug("Frame extraction options: interval_s={}, max_frames={}", interval, max_frames)

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    # TODO: Implement frame extraction with cv2.VideoCapture
    logger.warning("Frame extraction not yet implemented")


if __name__ == "__main__":
    main()
