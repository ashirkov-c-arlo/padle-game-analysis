# MVP Notes

## Player Visual Gap Fill

Annotated video and minimap smooth short player-track gaps visually.

- Enabled by `export.video.max_player_gap_fill_frames` in `configs/default.yaml`.
- Default: `15` frames.
- Method: linear interpolation by frame index.
- Bboxes in annotated video are linearly interpolated between neighboring real track observations.
- Player markers in minimap are linearly interpolated between neighboring real `PlayerMetricFrame.court_xy` observations.
- Interpolated visuals use reduced confidence styling.
- Source data stays unchanged: no synthetic rows are added to tracks, metrics, exports, or analytics.
