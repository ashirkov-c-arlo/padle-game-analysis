"""HTML dashboard generator for padel match analysis.

Generates a single self-contained HTML file with embedded Plotly charts
for presenting match analysis results. Uses Plotly CDN for rendering;
all data is embedded directly in the HTML.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger


def generate_dashboard(output_dir: Path) -> Path:
    """Generate an interactive HTML dashboard from pipeline output files.

    Args:
        output_dir: Directory containing pipeline CSV/JSON/JSONL outputs.

    Returns:
        Path to the generated dashboard.html file.
    """
    output_dir = Path(output_dir)
    data = _load_data(output_dir)
    html = _build_html(data)
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard generated: {}", dashboard_path)
    return dashboard_path


def _load_data(output_dir: Path) -> dict[str, Any]:
    """Load all available output files into a data dict."""
    data: dict[str, Any] = {}

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        data["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    geometry_path = output_dir / "court_geometry.json"
    if geometry_path.exists():
        data["geometry"] = json.loads(geometry_path.read_text(encoding="utf-8"))

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        data["metrics"] = _read_csv(metrics_path)

    ball_tracks_path = output_dir / "ball_tracks.csv"
    if ball_tracks_path.exists():
        data["ball_tracks"] = _read_csv(ball_tracks_path)

    events_path = output_dir / "ball_event_candidates.jsonl"
    if events_path.exists():
        data["ball_events"] = _read_jsonl(events_path)

    rally_path = output_dir / "rally_metrics.csv"
    if rally_path.exists():
        data["rally_metrics"] = _read_csv(rally_path)

    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV file into list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file into list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_html(data: dict[str, Any]) -> str:
    """Build the full HTML dashboard string."""
    summary = data.get("summary", {})
    geometry = data.get("geometry", {"width_m": 10.0, "length_m": 20.0, "net_y_m": 10.0})
    metrics = data.get("metrics", [])
    ball_events = data.get("ball_events", [])
    rally_metrics = data.get("rally_metrics", [])

    sections = []
    sections.append(_section_header(summary))
    sections.append(_section_player_heatmaps(metrics, geometry))
    sections.append(_section_bounce_map(ball_events, geometry))
    sections.append(_section_speed_timeline(metrics))
    sections.append(_section_zone_distribution(metrics))
    sections.append(_section_rally_stats(rally_metrics))
    sections.append(_section_shot_direction(ball_events))

    charts_js = "\n".join(s for s in sections if s)

    return _html_template(charts_js)


def _html_template(charts_js: str) -> str:
    """Wrap chart JS in full HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Padel Match Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 24px;
    line-height: 1.6;
}}
.dashboard-header {{
    text-align: center;
    padding: 32px 16px;
    margin-bottom: 32px;
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    border-radius: 12px;
    border: 1px solid #1a3a5c;
}}
.dashboard-header h1 {{
    font-size: 2.2rem;
    color: #ffffff;
    margin-bottom: 16px;
    letter-spacing: 1px;
}}
.stats-row {{
    display: flex;
    justify-content: center;
    gap: 32px;
    flex-wrap: wrap;
}}
.stat-card {{
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 12px 24px;
    text-align: center;
    min-width: 120px;
}}
.stat-card .value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #53c9f0;
}}
.stat-card .label {{
    font-size: 0.8rem;
    color: #9e9e9e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.section {{
    margin-bottom: 32px;
}}
.section h2 {{
    font-size: 1.3rem;
    color: #ffffff;
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 4px solid #53c9f0;
}}
.chart-container {{
    background: #16213e;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #1a3a5c;
}}
.chart-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 16px;
}}
.no-data {{
    text-align: center;
    padding: 40px;
    color: #666;
    font-style: italic;
}}
</style>
</head>
<body>
<div id="dashboard">
{charts_js}
</div>
<script>
// All charts rendered inline above
</script>
</body>
</html>"""


def _section_header(summary: dict) -> str:
    """Generate match summary header HTML."""
    video_name = Path(summary.get("video_path", "Unknown")).stem if summary.get("video_path") else "Unknown"
    duration = summary.get("duration_s", 0)
    total_frames = summary.get("total_frames", 0)
    fps = summary.get("fps", 0)
    player_count = len(summary.get("player_stats", {}))
    ball_stats = summary.get("ball_tracking", {})
    detection_rate = ball_stats.get("detection_rate", 0)

    duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}" if duration else "N/A"
    det_rate_str = f"{detection_rate * 100:.1f}%" if detection_rate else "N/A"

    return f"""
<div class="dashboard-header">
    <h1>Padel Match Analysis</h1>
    <div class="stats-row">
        <div class="stat-card"><div class="value">{video_name}</div><div class="label">Video</div></div>
        <div class="stat-card"><div class="value">{duration_str}</div><div class="label">Duration</div></div>
        <div class="stat-card"><div class="value">{total_frames}</div><div class="label">Frames</div></div>
        <div class="stat-card"><div class="value">{player_count}</div><div class="label">Players</div></div>
        <div class="stat-card"><div class="value">{det_rate_str}</div><div class="label">Ball Detection</div></div>
        <div class="stat-card"><div class="value">{fps:.1f}</div><div class="label">FPS</div></div>
    </div>
</div>
"""


def _court_shapes(geometry: dict) -> str:
    """Generate Plotly shapes JSON for court lines overlay."""
    w = geometry.get("width_m", 10.0)
    l = geometry.get("length_m", 20.0)
    net_y = geometry.get("net_y_m", 10.0)
    svc_offset = geometry.get("service_line_offset_from_net_m", 6.95)

    line_style = "'line': {'color': 'white', 'width': 2}"
    dash_style = "'line': {'color': 'rgba(200,200,200,0.7)', 'width': 2, 'dash': 'dash'}"

    shapes = []
    shapes.append(f"{{'type': 'rect', 'x0': 0, 'y0': 0, 'x1': {w}, 'y1': {l}, {line_style}, 'fillcolor': 'rgba(34,139,34,0.3)'}}")
    shapes.append(f"{{'type': 'line', 'x0': 0, 'y0': {net_y}, 'x1': {w}, 'y1': {net_y}, {dash_style}}}")
    near_svc = net_y - svc_offset
    far_svc = net_y + svc_offset
    thin_style = "'line': {'color': 'rgba(255,255,255,0.5)', 'width': 1}"
    shapes.append(f"{{'type': 'line', 'x0': 0, 'y0': {near_svc:.2f}, 'x1': {w}, 'y1': {near_svc:.2f}, {thin_style}}}")
    shapes.append(f"{{'type': 'line', 'x0': 0, 'y0': {far_svc:.2f}, 'x1': {w}, 'y1': {far_svc:.2f}, {thin_style}}}")
    center_x = w / 2
    shapes.append(f"{{'type': 'line', 'x0': {center_x}, 'y0': {near_svc:.2f}, 'x1': {center_x}, 'y1': {far_svc:.2f}, {thin_style}}}")

    return "[" + ",\n        ".join(shapes) + "]"


def _section_player_heatmaps(metrics: list[dict], geometry: dict) -> str:
    """Generate player heatmap section."""
    if not metrics:
        return _empty_section("Player Heatmaps", "No player metrics data available.")

    player_positions: dict[str, tuple[list[float], list[float]]] = {}
    for row in metrics:
        pid = row.get("player_id", "")
        try:
            x = float(row["court_x"])
            y = float(row["court_y"])
        except (ValueError, KeyError):
            continue
        if pid not in player_positions:
            player_positions[pid] = ([], [])
        player_positions[pid][0].append(x)
        player_positions[pid][1].append(y)

    if not player_positions:
        return _empty_section("Player Heatmaps", "No valid position data.")

    w = geometry.get("width_m", 10.0)
    l = geometry.get("length_m", 20.0)
    shapes_js = _court_shapes(geometry)

    chart_divs = []
    chart_scripts = []

    colors = {
        "near_left": "Blues",
        "near_right": "Teal",
        "far_left": "Reds",
        "far_right": "Oranges",
    }

    for pid, (xs, ys) in sorted(player_positions.items()):
        div_id = f"heatmap_{pid}"
        colorscale = colors.get(pid, "Viridis")
        team_label = "Near" if "near" in pid else "Far"
        side_label = "Left" if "left" in pid else "Right"

        chart_divs.append(f'<div class="chart-container"><div id="{div_id}"></div></div>')
        chart_scripts.append(f"""
<script>
(function() {{
    var xs = {json.dumps(xs)};
    var ys = {json.dumps(ys)};
    var trace = {{
        x: xs,
        y: ys,
        type: 'histogram2dcontour',
        colorscale: '{colorscale}',
        showscale: true,
        contours: {{coloring: 'heatmap'}},
        ncontours: 20,
        line: {{width: 0}},
        hoverinfo: 'z'
    }};
    var layout = {{
        title: {{text: '{team_label} {side_label} ({pid})', font: {{color: '#fff', size: 14}}}},
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: 'rgba(34,139,34,0.15)',
        xaxis: {{range: [0, {w}], title: 'Width (m)', dtick: 2.5, gridcolor: 'rgba(255,255,255,0.1)'}},
        yaxis: {{range: [0, {l}], title: 'Length (m)', dtick: 5, scaleanchor: 'x', gridcolor: 'rgba(255,255,255,0.1)'}},
        shapes: {shapes_js},
        margin: {{t: 40, b: 40, l: 50, r: 20}},
        height: 450
    }};
    Plotly.newPlot('{div_id}', [trace], layout, {{responsive: true}});
}})();
</script>""")

    divs_html = '\n'.join(chart_divs)
    scripts_html = '\n'.join(chart_scripts)

    return f"""
<div class="section">
    <h2>Player Heatmaps</h2>
    <div class="chart-row">
        {divs_html}
    </div>
</div>
{scripts_html}
"""


def _section_bounce_map(ball_events: list[dict], geometry: dict) -> str:
    """Generate bounce map scatter plot on court."""
    bounces = [e for e in ball_events if e.get("event_type") == "bounce_candidate" and e.get("court_xy_approx")]
    if not bounces:
        return _empty_section("Bounce Map", "No bounce events detected.")

    xs = [b["court_xy_approx"][0] for b in bounces]
    ys = [b["court_xy_approx"][1] for b in bounces]
    confs = [b.get("confidence", 0.5) for b in bounces]
    frames = [b.get("frame", 0) for b in bounces]

    w = geometry.get("width_m", 10.0)
    l = geometry.get("length_m", 20.0)
    shapes_js = _court_shapes(geometry)

    return f"""
<div class="section">
    <h2>Bounce Map</h2>
    <div class="chart-container"><div id="bounce_map"></div></div>
</div>
<script>
(function() {{
    var trace = {{
        x: {json.dumps(xs)},
        y: {json.dumps(ys)},
        mode: 'markers',
        type: 'scatter',
        marker: {{
            size: 12,
            color: {json.dumps(confs)},
            colorscale: 'YlOrRd',
            showscale: true,
            colorbar: {{title: 'Confidence', titlefont: {{color: '#ccc'}}, tickfont: {{color: '#ccc'}}}},
            line: {{color: 'white', width: 1}},
            opacity: 0.85
        }},
        text: {json.dumps([f"Frame {f}, Conf: {c:.2f}" for f, c in zip(frames, confs)])},
        hoverinfo: 'text+x+y'
    }};
    var layout = {{
        title: {{text: 'Ball Bounce Locations', font: {{color: '#fff', size: 14}}}},
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: 'rgba(34,139,34,0.15)',
        xaxis: {{range: [0, {w}], title: 'Width (m)', dtick: 2.5, gridcolor: 'rgba(255,255,255,0.1)'}},
        yaxis: {{range: [0, {l}], title: 'Length (m)', dtick: 5, scaleanchor: 'x', gridcolor: 'rgba(255,255,255,0.1)'}},
        shapes: {shapes_js},
        margin: {{t: 40, b: 40, l: 50, r: 20}},
        height: 500
    }};
    Plotly.newPlot('bounce_map', [trace], layout, {{responsive: true}});
}})();
</script>
"""


def _section_speed_timeline(metrics: list[dict]) -> str:
    """Generate speed over time line chart per player."""
    if not metrics:
        return _empty_section("Speed Timeline", "No player metrics data available.")

    player_data: dict[str, tuple[list[float], list[float]]] = {}
    for row in metrics:
        pid = row.get("player_id", "")
        try:
            t = float(row["time_s"])
            speed = float(row["speed_mps"])
        except (ValueError, KeyError):
            continue
        if pid not in player_data:
            player_data[pid] = ([], [])
        player_data[pid][0].append(t)
        player_data[pid][1].append(speed)

    if not player_data:
        return _empty_section("Speed Timeline", "No valid speed data.")

    colors_map = {
        "near_left": "#1f77b4",
        "near_right": "#17becf",
        "far_left": "#d62728",
        "far_right": "#ff7f0e",
    }

    traces = []
    for pid, (times, speeds) in sorted(player_data.items()):
        color = colors_map.get(pid, "#888888")
        traces.append(f"""{{
            x: {json.dumps(times)},
            y: {json.dumps(speeds)},
            type: 'scatter',
            mode: 'lines',
            name: '{pid}',
            line: {{color: '{color}', width: 1.5}},
            hovertemplate: '{pid}<br>Time: %{{x:.1f}}s<br>Speed: %{{y:.1f}} m/s<extra></extra>'
        }}""")

    traces_js = ",\n        ".join(traces)

    return f"""
<div class="section">
    <h2>Speed Timeline</h2>
    <div class="chart-container"><div id="speed_timeline"></div></div>
</div>
<script>
(function() {{
    var traces = [{traces_js}];
    var layout = {{
        title: {{text: 'Player Speed Over Time', font: {{color: '#fff', size: 14}}}},
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#16213e',
        xaxis: {{title: 'Time (s)', gridcolor: 'rgba(255,255,255,0.1)'}},
        yaxis: {{title: 'Speed (m/s)', gridcolor: 'rgba(255,255,255,0.1)'}},
        legend: {{font: {{color: '#ccc'}}}},
        margin: {{t: 40, b: 50, l: 60, r: 20}},
        height: 350
    }};
    Plotly.newPlot('speed_timeline', traces, layout, {{responsive: true}});
}})();
</script>
"""


def _section_zone_distribution(metrics: list[dict]) -> str:
    """Generate zone time distribution chart per player."""
    if not metrics:
        return _empty_section("Zone Distribution", "No player metrics data available.")

    zone_counts: dict[str, dict[str, int]] = {}
    for row in metrics:
        pid = row.get("player_id", "")
        zone = row.get("zone", "")
        if not pid or not zone:
            continue
        if pid not in zone_counts:
            zone_counts[pid] = {"net": 0, "mid": 0, "baseline": 0}
        if zone in zone_counts[pid]:
            zone_counts[pid][zone] += 1

    if not zone_counts:
        return _empty_section("Zone Distribution", "No zone data.")

    players = sorted(zone_counts.keys())
    net_pcts = []
    mid_pcts = []
    baseline_pcts = []

    for pid in players:
        counts = zone_counts[pid]
        total = sum(counts.values())
        if total == 0:
            net_pcts.append(0)
            mid_pcts.append(0)
            baseline_pcts.append(0)
        else:
            net_pcts.append(round(counts["net"] / total * 100, 1))
            mid_pcts.append(round(counts["mid"] / total * 100, 1))
            baseline_pcts.append(round(counts["baseline"] / total * 100, 1))

    return f"""
<div class="section">
    <h2>Zone Distribution</h2>
    <div class="chart-container"><div id="zone_dist"></div></div>
</div>
<script>
(function() {{
    var players = {json.dumps(players)};
    var trace_net = {{
        y: players,
        x: {json.dumps(net_pcts)},
        name: 'Net',
        type: 'bar',
        orientation: 'h',
        marker: {{color: '#2ecc71'}},
        hovertemplate: '%{{y}}<br>Net: %{{x:.1f}}%<extra></extra>'
    }};
    var trace_mid = {{
        y: players,
        x: {json.dumps(mid_pcts)},
        name: 'Mid',
        type: 'bar',
        orientation: 'h',
        marker: {{color: '#f39c12'}},
        hovertemplate: '%{{y}}<br>Mid: %{{x:.1f}}%<extra></extra>'
    }};
    var trace_baseline = {{
        y: players,
        x: {json.dumps(baseline_pcts)},
        name: 'Baseline',
        type: 'bar',
        orientation: 'h',
        marker: {{color: '#e74c3c'}},
        hovertemplate: '%{{y}}<br>Baseline: %{{x:.1f}}%<extra></extra>'
    }};
    var layout = {{
        title: {{text: 'Time in Zone (%)', font: {{color: '#fff', size: 14}}}},
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#16213e',
        barmode: 'stack',
        xaxis: {{title: 'Percentage (%)', range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)'}},
        yaxis: {{gridcolor: 'rgba(255,255,255,0.1)'}},
        legend: {{font: {{color: '#ccc'}}, orientation: 'h', y: -0.15}},
        margin: {{t: 40, b: 60, l: 100, r: 20}},
        height: 280
    }};
    Plotly.newPlot('zone_dist', [trace_net, trace_mid, trace_baseline], layout, {{responsive: true}});
}})();
</script>
"""


def _section_rally_stats(rally_metrics: list[dict]) -> str:
    """Generate rally duration bar chart."""
    if not rally_metrics:
        return _empty_section("Rally Stats", "No rally data detected.")

    rally_ids = []
    durations = []
    shots = []

    for row in rally_metrics:
        try:
            rid = row.get("rally_id", "")
            dur = float(row.get("duration_s", 0))
            est_shots = int(row.get("estimated_shots", 0))
        except (ValueError, TypeError):
            continue
        rally_ids.append(f"Rally {rid}")
        durations.append(round(dur, 2))
        shots.append(est_shots)

    if not durations:
        return _empty_section("Rally Stats", "No valid rally data.")

    return f"""
<div class="section">
    <h2>Rally Stats</h2>
    <div class="chart-container"><div id="rally_stats"></div></div>
</div>
<script>
(function() {{
    var rallies = {json.dumps(rally_ids)};
    var durations = {json.dumps(durations)};
    var shots = {json.dumps(shots)};
    var text_labels = shots.map(function(s) {{ return s + ' shots'; }});
    var trace = {{
        x: rallies,
        y: durations,
        type: 'bar',
        marker: {{
            color: durations,
            colorscale: 'Viridis',
            showscale: false
        }},
        text: text_labels,
        textposition: 'outside',
        textfont: {{color: '#ccc', size: 11}},
        hovertemplate: '%{{x}}<br>Duration: %{{y:.1f}}s<br>Shots: %{{text}}<extra></extra>'
    }};
    var layout = {{
        title: {{text: 'Rally Duration & Shot Count', font: {{color: '#fff', size: 14}}}},
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#16213e',
        xaxis: {{gridcolor: 'rgba(255,255,255,0.1)'}},
        yaxis: {{title: 'Duration (s)', gridcolor: 'rgba(255,255,255,0.1)'}},
        margin: {{t: 40, b: 50, l: 60, r: 20}},
        height: 320
    }};
    Plotly.newPlot('rally_stats', [trace], layout, {{responsive: true}});
}})();
</script>
"""


def _section_shot_direction(ball_events: list[dict]) -> str:
    """Generate shot direction pie chart from touch events."""
    touches = [e for e in ball_events if e.get("event_type") == "touch_candidate"]
    if not touches:
        return _empty_section("Shot Direction", "No touch event data available for shot direction analysis.")

    directions = {"cross_court": 0, "down_the_line": 0, "middle": 0}
    court_width = 10.0
    classified = 0

    for touch in touches:
        x = None
        if touch.get("court_xy_approx"):
            x = touch["court_xy_approx"][0]
        elif touch.get("image_xy"):
            img_x = touch["image_xy"][0]
            x = (img_x / 1920.0) * court_width
            x = max(0.0, min(court_width, x))

        if x is None:
            continue

        third = court_width / 3
        if x < third:
            directions["down_the_line"] += 1
        elif x > court_width - third:
            directions["cross_court"] += 1
        else:
            directions["middle"] += 1
        classified += 1

    if classified == 0:
        return _empty_section("Shot Direction", "No classifiable shot data.")

    total = sum(directions.values())
    if total == 0:
        return _empty_section("Shot Direction", "No classifiable shots.")

    labels = ["Cross Court", "Down the Line", "Middle"]
    values = [directions["cross_court"], directions["down_the_line"], directions["middle"]]
    colors = ["#3498db", "#e74c3c", "#f39c12"]

    return f"""
<div class="section">
    <h2>Shot Direction</h2>
    <div class="chart-container"><div id="shot_direction"></div></div>
</div>
<script>
(function() {{
    var trace = {{
        labels: {json.dumps(labels)},
        values: {json.dumps(values)},
        type: 'pie',
        hole: 0.4,
        marker: {{colors: {json.dumps(colors)}}},
        textinfo: 'label+percent',
        textfont: {{color: '#fff'}},
        hovertemplate: '%{{label}}<br>Count: %{{value}}<br>%{{percent}}<extra></extra>'
    }};
    var layout = {{
        title: {{text: 'Shot Direction Distribution', font: {{color: '#fff', size: 14}}}},
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#16213e',
        showlegend: true,
        legend: {{font: {{color: '#ccc'}}}},
        margin: {{t: 40, b: 20, l: 20, r: 20}},
        height: 350
    }};
    Plotly.newPlot('shot_direction', [trace], layout, {{responsive: true}});
}})();
</script>
"""


def _empty_section(title: str, message: str) -> str:
    """Generate a placeholder section when data is missing."""
    return f"""
<div class="section">
    <h2>{title}</h2>
    <div class="chart-container">
        <div class="no-data">{message}</div>
    </div>
</div>
"""
