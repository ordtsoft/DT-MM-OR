# Minimal shapefile viewer

An animated viewer for the numbered `camera01_centroids_*.shp` files. It draws
the classified point colors and names, a meter grid, labeled axes, and a stable
view across the sequence. It also reads the matching per-frame JSON files from
`relation_labels`, draws directional relations between visible entities, and
lists every relation in a side panel. Both grid axes begin at zero.

Source RGB colors are displayed as:

- `(10, 0, 0)`: `instrument_table`, RGB `(255, 51, 153)`
- `(170, 0, 0)`: `robot`, RGB `(60, 75, 255)`
- `(110, 0, 0)`: `robot technician (mps)`, RGB `(125, 100, 25)`
- `(40, 0, 0)`: `mps station`, RGB `(133, 0, 133)`
- `(80, 0, 0)`: `circulator`, RGB `(255, 128, 0)`
- `(120, 0, 0)`: `nurse`, RGB `(128, 255, 0)`
- `(70, 0, 0)`: `anesthesist`, RGB `(177, 255, 110)`
- `(90, 0, 0)`: `assistant surgeon`, RGB `(116, 166, 116)`
- `(100, 0, 0)`: `head surgeon`, RGB `(76, 161, 245)`
- `(50, 0, 0)`: `patient`, RGB `(255, 0, 0)`
- `(60, 0, 0)`: `drape`, RGB `(183, 91, 255)`
- `(130, 0, 0)`: `drill`, RGB `(0, 255, 128)`
- `(30, 0, 0)`: `operating table`, RGB `(255, 255, 0)`

## Run

```powershell
python -m pip install -r requirements.txt
python viewer.py
```

Use **Play**, the slider, the arrow keys, or the space bar. A different folder
or filename pattern can be supplied if needed. Relation JSON files can be
selected separately with `--relations`:

```powershell
python viewer.py "C:\path\to\shapefiles" --pattern "*.shp" --relations "C:\path\to\relation_labels" --interval 80
```

Dashed lines mean proximity and solid arrows mean an interaction. The supplied
annotations do not contain a literal sterility-breach class, so the viewer
infers a breach when a germ carrier participates in direct contact (for example
`Touching`, `Holding`, or `Manipulating`) with a clean entity. `CloseTo` never
transmits a germ. A green germ marks the transmission frame; contamination then
persists, and any visible contaminated entity receives a red spiked contour.

Only Python, Tk (normally included with Python), and the small `pyshp` package
are required.

## Tests

Install the development dependencies and run the headless test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests cover numeric frame ordering, motion-model fitting, predictor cold
starts and warm-up, missing detections, and causal one-, two-, and multi-frame
prediction ordering.

## Real-time MQTT prototype

An isolated real-time experiment is available in
[`realtime_mqtt`](realtime_mqtt/README.md). It supports a broker-free loopback
demo plus MQTT publishing/receiving with configurable latency, jitter, dropped
frames, corruption, duplicates, and out-of-order delivery.

## Cold-start trajectory prediction

`viewer_prediction.py` is a separate, shapefile-only viewer. It does not load
relation annotations or interaction/infection state. It initially displays only
`instrument_table` and predicts that entity's motion using earlier observations
only:

```powershell
python viewer_prediction.py
```

The blue path is the observed trajectory. The dashed orange path contains the
one-step predictions made before each observation arrived, and the orange arrow
forecasts the next frames. With one sample the cold-start estimate is stationary;
as observations arrive, a rolling online regression learns local X/Y velocity.
The model advances lazily during playback, so startup does not scan the entire
sequence.

Choose another single entity or tune the history and forecast lengths with:

```powershell
python viewer_prediction.py --entity patient --window 45 --trail 100 --horizon 30
```

## Predictor comparison on a moving entity

`viewer_comparison.py` defaults to `circulator`, the most mobile well-observed
entity in the first 638 frames (about 84.3 m of cumulative path). It compares
seven cold-start, causal algorithms on exactly the same observations:

- Persistence (last known position)
- Exponentially smoothed constant velocity
- Rolling linear regression
- Alpha-beta position/velocity filter
- Online neural network trained incrementally by back-propagation
- Memory-based k-nearest neighbors with a 40-example warm-up
- Adaptive expert ensemble that learns when velocity is safer than persistence

```powershell
python viewer_comparison.py
```

Only the entity's actual position for the selected frame is shown; the black
trajectory trail is hidden. Each algorithm is represented by one colored ring
at its prediction for that same frame. The prediction was made before that
frame's observation arrived, so it is exactly the value used by the current
error metric. Use the **Plot** selector to isolate one model when rings overlap.
Use the mouse wheel over
the plot for cursor-centered zoom, or use the `−`, `100%`, and `+` buttons.

A live side panel ranks the methods by one-step mean absolute error and also
reports RMSE and current error. Missing entity detections are excluded
consistently from every method. As with the single-model viewer, no relation or
interaction data is loaded and unseen frames are processed lazily. The neural
network begins with reproducibly poor random weights and reports its
training-example count as it learns; no pre-training or future frames are used.

See [TRAJECTORY_ALGORITHMS.md](TRAJECTORY_ALGORITHMS.md) for a short explanation
of each method and metric.

## Strict two-frame prediction

`viewer_two_frame.py` runs the same comparison at a harder two-frame horizon:

```powershell
python viewer_two_frame.py
```

A prediction scored at frame `t` is generated and frozen immediately after
frame `t-2`; neither intervening observation is available to it. Frames zero
and one therefore have no prediction score. Playback and slider jumps retain
this causal ordering, and the side panel reports two-frame-ahead errors.

## Clear multi-frame prediction

`viewer_multi_frame.py` defaults to a 10-frame horizon and keeps only three
models for visual clarity: persistence, the online neural network, and the
adaptive ensemble.

```powershell
python viewer_multi_frame.py
```

The selected frame shows its actual position and one colored prediction ring
per model. Each displayed prediction was frozen ten frames earlier and is the
value currently being scored. There are no trajectory trails, future markers,
predicted-path lines, or arrows. Change the horizon when needed:

```powershell
python viewer_multi_frame.py --steps 20
```

Predictions remain frozen until their target frames arrive, so increasing the
horizon never exposes intervening observations.
