from __future__ import annotations

import numpy as np

from src.schemas import CourtGeometry2D, FormationState, PlayerID, Zone


def classify_zone(court_xy: tuple[float, float], geometry: CourtGeometry2D) -> Zone:
    """Classify player position into zone based on distance from net.

    Court layout:
    - y=0 is far baseline, y=10 is net, y=20 is near baseline
    - Near team players: y > 10, Far team players: y < 10
    - Distance from net = abs(y - 10) for any player

    Zone boundaries (distance from net):
    - net: 0.0 - zone_net_distance_m (3.5m)
    - mid: zone_net_distance_m - zone_mid_distance_m (3.5 - 6.95m)
    - baseline: zone_mid_distance_m - 10.0m (6.95 - 10.0m)

    Args:
        court_xy: (x, y) position in court coordinates (meters).
        geometry: Court geometry with zone boundary distances.

    Returns:
        Zone classification: "net", "mid", or "baseline".
    """
    _, y = court_xy
    distance_from_net = abs(y - geometry.net_y_m)

    if distance_from_net <= geometry.zone_net_distance_m:
        return "net"
    elif distance_from_net <= geometry.zone_mid_distance_m:
        return "mid"
    else:
        return "baseline"


def compute_zone_time(
    positions: list[tuple[float, float]],
    player_id: PlayerID,
    geometry: CourtGeometry2D,
    fps: float,
) -> dict[str, float]:
    """Compute fraction of time spent in each zone.

    Args:
        positions: List of (x, y) court coordinates for the player.
        player_id: Player identifier (unused in computation but kept for interface).
        geometry: Court geometry with zone boundaries.
        fps: Video frame rate (unused, each frame counts equally).

    Returns:
        Dictionary with zone fractions: {"net": f, "mid": f, "baseline": f}.
        Fractions sum to 1.0. Returns zeros if no positions.
    """
    if not positions:
        return {"net": 0.0, "mid": 0.0, "baseline": 0.0}

    zone_counts: dict[str, int] = {"net": 0, "mid": 0, "baseline": 0}

    for pos in positions:
        zone = classify_zone(pos, geometry)
        zone_counts[zone] += 1

    total = len(positions)
    return {
        "net": zone_counts["net"] / total,
        "mid": zone_counts["mid"] / total,
        "baseline": zone_counts["baseline"] / total,
    }


def classify_formation(
    player1_zone: Zone,
    player2_zone: Zone,
) -> FormationState:
    """Classify team formation based on both players' zones.

    Formation rules:
    - both_net: both in net zone
    - both_mid: both in mid zone
    - both_baseline: both in baseline zone
    - one_up_one_back: one in net + one in baseline, or one in net + one in mid,
      or one in mid + one in baseline
    - split_unknown: should not occur with 3 zones, but kept for safety

    Args:
        player1_zone: Zone classification for player 1.
        player2_zone: Zone classification for player 2.

    Returns:
        FormationState describing the team formation.
    """
    zones = {player1_zone, player2_zone}

    if player1_zone == player2_zone:
        if player1_zone == "net":
            return "both_net"
        elif player1_zone == "mid":
            return "both_mid"
        elif player1_zone == "baseline":
            return "both_baseline"

    # Different zones = one up one back
    if len(zones) == 2:
        return "one_up_one_back"

    return "split_unknown"


def compute_partner_spacing(
    pos1: tuple[float, float],
    pos2: tuple[float, float],
) -> float:
    """Euclidean distance between two teammates in court coords.

    Args:
        pos1: (x, y) court position of player 1.
        pos2: (x, y) court position of player 2.

    Returns:
        Distance in meters between the two players.
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return float(np.sqrt(dx * dx + dy * dy))
