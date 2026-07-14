"""Minimal animated viewer for a numbered sequence of point shapefiles."""

from __future__ import annotations

import argparse
import json
import math
import re
import tkinter as tk
from functools import lru_cache
from pathlib import Path
from tkinter import messagebox, ttk

import shapefile


BACKGROUND = "#f8fafc"
GRID = "#dbe3ec"
AXIS = "#52606d"
FALLBACK_COLORS = ("#e11d48", "#ea580c", "#ca8a04", "#16a34a", "#0891b2", "#4f46e5", "#9333ea")
RELATION_COLORS = {
    "CloseTo": "#64748b", "Touching": "#7c3aed",
    "Holding": "#0284c7", "Manipulating": "#0f766e",
}
GERM_COLOR = "#65a30d"
INFECTION_COLOR = "#dc2626"

# Relation annotation names -> display-layer names.
ENTITY_ALIASES = {
    "anest": "anesthesist", "mako_robot": "robot",
    "mps": "robot technician (mps)", "ot": "operating table",
    "assistant_surgeon": "assistant surgeon", "head_surgeon": "head surgeon",
    "instrument_table": "instrument_table", "mps_station": "mps station",
}
DISPLAY_TO_ENTITY = {display: entity for entity, display in ENTITY_ALIASES.items()}
DISPLAY_TO_ENTITY.update({
    "circulator": "circulator", "drape": "drape", "drill": "drill",
    "nurse": "nurse", "patient": "patient",
})

# These roles are possible initial germ carriers. Transmission requires an
# annotated direct contact; CloseTo is intentionally not sufficient.
INITIAL_GERM_CARRIERS = frozenset({"anest", "circulator", "mps"})
DIRECT_CONTACT_RELATIONS = frozenset({
    "touching", "holding", "manipulating", "preparing", "drilling",
    "sawing", "suturing", "hammering", "scanning", "calibrating",
})

# Source RGB -> (display name, display color)
COLOR_CLASSES = {
    (10, 0, 0): ("instrument_table", "#FF3399"),
    (30, 0, 0): ("operating table", "#FFFF00"),
    (40, 0, 0): ("mps station", "#850085"),
    (50, 0, 0): ("patient", "#FF0000"),
    (60, 0, 0): ("drape", "#B75BFF"),
    (70, 0, 0): ("anesthesist", "#B1FF6E"),
    (80, 0, 0): ("circulator", "#FF8000"),
    (90, 0, 0): ("assistant surgeon", "#74A674"),
    (100, 0, 0): ("head surgeon", "#4CA1F5"),
    (110, 0, 0): ("robot technician (mps)", "#7D6419"),
    (120, 0, 0): ("nurse", "#80FF00"),
    (130, 0, 0): ("drill", "#00FF80"),
    (170, 0, 0): ("robot", "#3C4BFF"),
}


def sequence_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def nice_step(span: float, target_lines: int = 8) -> float:
    raw = max(span / target_lines, 1e-12)
    power = 10 ** math.floor(math.log10(raw))
    fraction = raw / power
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice * power


class ShapeSequence:
    def __init__(self, directory: Path, pattern: str, relation_directory: Path | None = None) -> None:
        self.files = sorted(directory.glob(pattern), key=sequence_number)
        if not self.files:
            raise FileNotFoundError(f"No shapefiles match {pattern!r} in {directory}")
        self.relation_directory = relation_directory
        self._relations = self._load_relations()
        self._contamination, self._transmissions = self._build_transmission_history()
        self.bounds = self._sampled_bounds()

    def _load_relations(self) -> list[list[tuple[str, str, str]]]:
        frames: list[list[tuple[str, str, str]]] = []
        for index in range(len(self.files)):
            path = self.relation_directory / f"{index:06d}.json" if self.relation_directory else None
            relations: list[tuple[str, str, str]] = []
            if path and path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for value in payload.get("rel_annotations", []):
                        if isinstance(value, list) and len(value) == 3 and all(isinstance(part, str) for part in value):
                            relations.append(tuple(value))
                except (OSError, json.JSONDecodeError):
                    # Optional relation data should never hide its shape frame.
                    pass
            frames.append(relations)
        return frames

    def _build_transmission_history(self) -> tuple[list[frozenset[str]], list[list[tuple[str, str, str]]]]:
        contaminated = set(INITIAL_GERM_CARRIERS) if any(self._relations) else set()
        history: list[frozenset[str]] = []
        events: list[list[tuple[str, str, str]]] = []
        for relations in self._relations:
            frame_events: list[tuple[str, str, str]] = []
            for source, relation, target in relations:
                normalized = relation.lower().replace("_", "").replace(" ", "")
                explicit_breach = "sterility" in relation.lower() and "breach" in relation.lower()
                if not (explicit_breach or normalized in DIRECT_CONTACT_RELATIONS):
                    continue
                source_dirty, target_dirty = source in contaminated, target in contaminated
                if explicit_breach:
                    if source_dirty != target_dirty:
                        carrier, recipient = (source, target) if source_dirty else (target, source)
                    else:
                        # The annotation itself establishes a breach even when
                        # no prior carrier state tells us its direction.
                        carrier, recipient = source, target
                    contaminated.update((carrier, recipient))
                    frame_events.append((carrier, relation, recipient))
                    continue
                if source_dirty != target_dirty:
                    carrier, recipient = (source, target) if source_dirty else (target, source)
                    contaminated.add(recipient)
                    frame_events.append((carrier, relation, recipient))
            history.append(frozenset(contaminated))
            events.append(frame_events)
        return history, events

    def relations(self, index: int) -> list[tuple[str, str, str]]:
        return self._relations[index]

    def contamination(self, index: int) -> frozenset[str]:
        return self._contamination[index]

    def transmissions(self, index: int) -> list[tuple[str, str, str]]:
        return self._transmissions[index]

    def _sampled_bounds(self) -> tuple[float, float, float, float]:
        """Estimate stable plot bounds quickly without opening thousands of files."""
        last = len(self.files) - 1
        indices = sorted({round(i * last / min(last, 59)) for i in range(min(len(self.files), 60))}) if last else [0]
        boxes = []
        for index in indices:
            with shapefile.Reader(str(self.files[index])) as reader:
                if len(reader):
                    boxes.append(tuple(reader.bbox))
        if not boxes:
            return (0.0, 0.0, 1.0, 1.0)
        return (
            min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes),
        )

    @lru_cache(maxsize=80)
    def frame(self, index: int) -> tuple[list[dict], int | None]:
        points: list[dict] = []
        frame_id = None
        with shapefile.Reader(str(self.files[index])) as reader:
            for position, item in enumerate(reader.iterShapeRecords()):
                if not item.shape.points:
                    continue
                attributes = item.record.as_dict()
                frame_id = attributes.get("FRAME_ID", frame_id)
                source_rgb = (
                    attributes.get("R"),
                    attributes.get("G"),
                    attributes.get("B"),
                )
                display_name, display_color = COLOR_CLASSES.get(
                    source_rgb,
                    (None, attributes.get("RGB_HEX") or FALLBACK_COLORS[position % len(FALLBACK_COLORS)]),
                )
                points.append({
                    "x": item.shape.points[0][0],
                    "y": item.shape.points[0][1],
                    "color": display_color,
                    "name": display_name,
                    "entity": DISPLAY_TO_ENTITY.get(display_name),
                    "interpolated": bool(attributes.get("INTERP", False)),
                    "pixels": attributes.get("N_PX"),
                })
        return points, frame_id


class Viewer(tk.Tk):
    def __init__(self, sequence: ShapeSequence, interval_ms: int) -> None:
        super().__init__()
        self.sequence = sequence
        self.interval_ms = interval_ms
        self.playing = False
        self.after_id: str | None = None
        self.title("Shapefile sequence viewer")
        self.geometry("1000x720")
        self.minsize(620, 440)

        self.canvas = tk.Canvas(self, background=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

        controls = ttk.Frame(self, padding=(10, 7))
        controls.pack(fill="x")
        ttk.Button(controls, text="◀", width=4, command=lambda: self.move(-1)).pack(side="left")
        self.play_button = ttk.Button(controls, text="Play", width=7, command=self.toggle_play)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(controls, text="▶", width=4, command=lambda: self.move(1)).pack(side="left")

        self.frame_var = tk.IntVar(value=1)
        self.slider = ttk.Scale(
            controls, from_=1, to=len(sequence.files), variable=self.frame_var,
            command=self.on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=12)
        self.status = ttk.Label(controls, width=34, anchor="e")
        self.status.pack(side="right")

        self.bind("<space>", lambda _event: self.toggle_play())
        self.bind("<Left>", lambda _event: self.move(-1))
        self.bind("<Right>", lambda _event: self.move(1))
        self.bind("<Home>", lambda _event: self.set_frame(0))
        self.bind("<End>", lambda _event: self.set_frame(len(self.sequence.files) - 1))
        self.after_idle(self.draw)

    @property
    def index(self) -> int:
        return max(0, min(len(self.sequence.files) - 1, round(self.frame_var.get()) - 1))

    def on_slider(self, _value: str) -> None:
        self.draw()

    def set_frame(self, index: int) -> None:
        self.frame_var.set(index + 1)
        self.draw()

    def move(self, amount: int) -> None:
        self.set_frame((self.index + amount) % len(self.sequence.files))

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()
        elif self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

    def _tick(self) -> None:
        if not self.playing:
            return
        self.move(1)
        self.after_id = self.after(self.interval_ms, self._tick)

    def draw(self) -> None:
        if self.canvas.winfo_width() < 20 or self.canvas.winfo_height() < 20:
            return
        try:
            points, frame_id = self.sequence.frame(self.index)
        except Exception as exc:
            self.playing = False
            self.play_button.configure(text="Play")
            messagebox.showerror("Could not read shapefile", str(exc))
            return

        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        panel_w = 270 if width >= 800 else 190
        left, right, top, bottom = 78, panel_w + 24, 30, 62
        _, _, x1, y1 = self.sequence.bounds
        # These camera coordinates are positive; anchor the grid at the origin
        # so low-valued points are not drawn beyond an auto-fitted border.
        x0, y0 = 0.0, 0.0
        x1 = max(x1, 0.1) * 1.06
        y1 = max(y1, 0.1) * 1.06
        plot_w, plot_h = max(width - left - right, 1), max(height - top - bottom, 1)

        def screen(x: float, y: float) -> tuple[float, float]:
            return left + (x - x0) / (x1 - x0) * plot_w, top + (y1 - y) / (y1 - y0) * plot_h

        x_step, y_step = nice_step(x1 - x0), nice_step(y1 - y0)
        x_tick = math.ceil(x0 / x_step) * x_step
        while x_tick <= x1 + x_step * 1e-6:
            sx, _ = screen(x_tick, y0)
            self.canvas.create_line(sx, top, sx, top + plot_h, fill=GRID)
            self.canvas.create_text(sx, top + plot_h + 18, text=f"{x_tick:g}", fill=AXIS, font=("Segoe UI", 9))
            x_tick += x_step
        y_tick = math.ceil(y0 / y_step) * y_step
        while y_tick <= y1 + y_step * 1e-6:
            _, sy = screen(x0, y_tick)
            self.canvas.create_line(left, sy, left + plot_w, sy, fill=GRID)
            self.canvas.create_text(left - 12, sy, text=f"{y_tick:g}", anchor="e", fill=AXIS, font=("Segoe UI", 9))
            y_tick += y_step

        self.canvas.create_rectangle(left, top, left + plot_w, top + plot_h, outline=AXIS)
        self.canvas.create_text(left + plot_w / 2, height - 17, text="X (m)", fill=AXIS, font=("Segoe UI", 10))
        self.canvas.create_text(18, top + plot_h / 2, text="Y (m)", angle=90, fill=AXIS, font=("Segoe UI", 10))

        relations = self.sequence.relations(self.index)
        transmissions = self.sequence.transmissions(self.index)
        contaminated = self.sequence.contamination(self.index)
        transmission_pairs = {(carrier, recipient) for carrier, _, recipient in transmissions}
        point_by_entity = {point["entity"]: point for point in points if point["entity"]}

        # Relations sit beneath nodes. The panel also preserves annotations for
        # entities (such as a saw or tracker) absent from the point layer.
        for relation_index, (source, relation, target) in enumerate(relations):
            source_point = point_by_entity.get(source)
            target_point = point_by_entity.get(target)
            if not source_point or not target_point:
                continue
            sx, sy = screen(source_point["x"], source_point["y"])
            tx, ty = screen(target_point["x"], target_point["y"])
            color = RELATION_COLORS.get(relation, "#475569")
            self.canvas.create_line(
                sx, sy, tx, ty, fill=color, width=2,
                dash=(5, 4) if relation.lower() == "closeto" else (),
                arrow="last", arrowshape=(7, 8, 3),
            )
            mx, my = (sx + tx) / 2, (sy + ty) / 2 + (relation_index % 3 - 1) * 10
            self.canvas.create_text(mx, my, text=relation, fill=color, font=("Segoe UI Semibold", 8))
            if (source, target) in transmission_pairs or (target, source) in transmission_pairs:
                self.draw_germ(mx, my - 11)

        for point in points:
            sx, sy = screen(point["x"], point["y"])
            radius = 7
            outline = "#111827" if point["interpolated"] else "white"
            if point["entity"] in contaminated:
                self.draw_spiky_contour(sx, sy, radius + 5)
            self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius,
                                    fill=point["color"], outline=outline, width=2)
            if point["name"]:
                state = " (infected)" if point["entity"] in contaminated else ""
                self.canvas.create_text(
                    sx + radius + 5, sy, text=point["name"] + state, anchor="w",
                    fill="#17202a", font=("Segoe UI", 9),
                )

        self.draw_relation_panel(left + plot_w + 18, top, panel_w, relations, transmissions)

        filename = self.sequence.files[self.index].name
        id_text = f" • source frame {frame_id}" if frame_id is not None else ""
        interpolated = sum(point["interpolated"] for point in points)
        self.status.configure(text=f"{self.index + 1} / {len(self.sequence.files)} • {len(points)} points{id_text}")
        self.canvas.create_text(left, 13, anchor="w", text=filename, fill="#17202a", font=("Segoe UI Semibold", 11))
        self.canvas.create_text(left + plot_w, 13, anchor="e",
                                text=f"{interpolated} interpolated" if interpolated else "observed",
                                fill=AXIS, font=("Segoe UI", 9))

    def draw_spiky_contour(self, x: float, y: float, radius: float) -> None:
        vertices = []
        for index in range(24):
            angle = index * math.pi / 12
            distance = radius + 4 if index % 2 == 0 else radius
            vertices.extend((x + math.cos(angle) * distance, y + math.sin(angle) * distance))
        self.canvas.create_polygon(vertices, fill="", outline=INFECTION_COLOR, width=2)

    def draw_germ(self, x: float, y: float, radius: float = 6) -> None:
        vertices = []
        for index in range(16):
            angle = index * math.pi / 8
            distance = radius + 3 if index % 2 == 0 else radius
            vertices.extend((x + math.cos(angle) * distance, y + math.sin(angle) * distance))
        self.canvas.create_polygon(vertices, fill="#a3e635", outline=GERM_COLOR, width=1)
        self.canvas.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill=GERM_COLOR, outline="")

    def draw_relation_panel(
        self, x: float, y: float, width: float,
        relations: list[tuple[str, str, str]], transmissions: list[tuple[str, str, str]],
    ) -> None:
        transmission_pairs = {(carrier, recipient) for carrier, _, recipient in transmissions}
        self.canvas.create_text(x, y, anchor="nw", text="Entity relations", fill="#17202a",
                                font=("Segoe UI Semibold", 11))
        self.canvas.create_text(x, y + 20, anchor="nw", text="dashed = proximity  •  arrow = direction",
                                fill=AXIS, font=("Segoe UI", 8))
        row_y = y + 45
        if not relations:
            self.canvas.create_text(x, row_y, anchor="nw", text="No annotations for this frame",
                                    fill=AXIS, font=("Segoe UI", 9))
        for source, relation, target in relations[:18]:
            color = RELATION_COLORS.get(relation, "#475569")
            breached = (source, target) in transmission_pairs or (target, source) in transmission_pairs
            self.canvas.create_oval(x, row_y + 4, x + 7, row_y + 11, fill=color, outline="")
            self.canvas.create_text(
                x + 12, row_y, anchor="nw", text=f"{source}  {relation}  {target}",
                width=max(width - 20, 80), fill=INFECTION_COLOR if breached else "#334155",
                font=("Segoe UI Semibold" if breached else "Segoe UI", 8),
            )
            if breached:
                self.draw_germ(x + 3.5, row_y + 22, 4)
                self.canvas.create_text(x + 12, row_y + 16, anchor="nw",
                                        text="STERILITY BREACH: germ transmitted",
                                        fill=INFECTION_COLOR, font=("Segoe UI Semibold", 8))
                row_y += 18
            row_y += 31
        if len(relations) > 18:
            self.canvas.create_text(x, row_y, anchor="nw", text=f"+ {len(relations) - 18} more",
                                    fill=AXIS, font=("Segoe UI", 8))

        legend_y = max(row_y + 15, self.canvas.winfo_height() - 92)
        self.draw_germ(x + 6, legend_y + 5, 4)
        self.canvas.create_text(x + 17, legend_y, anchor="nw", text="germ transmission",
                                fill=GERM_COLOR, font=("Segoe UI", 8))
        self.draw_spiky_contour(x + 6, legend_y + 27, 4)
        self.canvas.create_text(x + 17, legend_y + 22, anchor="nw", text="infected / contaminated",
                                fill=INFECTION_COLOR, font=("Segoe UI", 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent,
                        help="directory containing the shapefiles")
    parser.add_argument("--pattern", default="camera01_centroids_*.shp", help="shapefile glob pattern")
    parser.add_argument("--relations", type=Path, help="directory containing per-frame relation JSON files")
    parser.add_argument("--interval", type=int, default=100, help="animation interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    if args.directory == Path(__file__).parent and (directory / "shp").is_dir():
        directory = directory / "shp"
    relation_directory = args.relations.resolve() if args.relations else directory.parent / "relation_labels"
    if not relation_directory.is_dir():
        relation_directory = None
    try:
        sequence = ShapeSequence(directory, args.pattern, relation_directory)
    except (FileNotFoundError, shapefile.ShapefileException) as exc:
        raise SystemExit(str(exc)) from exc
    Viewer(sequence, max(args.interval, 10)).mainloop()


if __name__ == "__main__":
    main()
