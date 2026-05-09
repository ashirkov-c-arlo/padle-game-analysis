"""Train ball tracker model."""
from __future__ import annotations

import click
from loguru import logger

from src.logging_config import LOG_LEVELS, configure_logging


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Training data path")
@click.option("--out", default="data/models", type=click.Path(), help="Model output directory")
@click.option("--epochs", default=100, type=int, help="Number of training epochs")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log level. Defaults to PADEL_CV_LOG_LEVEL or INFO.",
)
def main(data: str, out: str, epochs: int, log_level: str | None) -> None:
    """Train the ball tracker model."""
    configure_logging(log_level)
    logger.info("Training ball tracker: data={}, output={}", data, out)
    logger.debug("Training options: epochs={}", epochs)

    # TODO: Implement ball tracker training
    logger.warning("Ball tracker training not yet implemented")


if __name__ == "__main__":
    main()
