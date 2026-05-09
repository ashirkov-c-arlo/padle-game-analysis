"""Train scoreboard detector model."""
from __future__ import annotations

import click
from loguru import logger


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Training data path")
@click.option("--out", default="data/models", type=click.Path(), help="Model output directory")
@click.option("--epochs", default=50, type=int, help="Number of training epochs")
def main(data: str, out: str, epochs: int) -> None:
    """Train the scoreboard detection model."""
    logger.info(f"Training scoreboard detector on {data}")
    logger.info(f"Epochs: {epochs}, Output: {out}")

    # TODO: Implement scoreboard detector training
    logger.warning("Scoreboard detector training not yet implemented")


if __name__ == "__main__":
    main()
