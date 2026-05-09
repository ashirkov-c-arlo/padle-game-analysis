"""Train ball tracker model."""
from __future__ import annotations

import click
from loguru import logger


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Training data path")
@click.option("--out", default="data/models", type=click.Path(), help="Model output directory")
@click.option("--epochs", default=100, type=int, help="Number of training epochs")
def main(data: str, out: str, epochs: int) -> None:
    """Train the ball tracker model."""
    logger.info(f"Training ball tracker on {data}")
    logger.info(f"Epochs: {epochs}, Output: {out}")

    # TODO: Implement ball tracker training
    logger.warning("Ball tracker training not yet implemented")


if __name__ == "__main__":
    main()
