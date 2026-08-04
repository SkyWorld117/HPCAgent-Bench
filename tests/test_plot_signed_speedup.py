# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_speedup.py`` -- the signed-change speed-up chart.

The load-bearing assertions are about the AXIS, not the drawing. A 2x speed-up and a 2x
slow-down must be the same distance from 0 (the whole reason the figure replaces a ratio axis),
and a cell that cannot be turned into a speed-up must not be able to land on 0, which is the exact
value of "measured, and nothing changed". Both are pure functions, so both are tested without
rendering anything.
"""
import importlib.util
import math
import pathlib
from typing import List

import pandas as pd
import pytest

from hpcagent_bench import plotting

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    """Import ``scripts/plot_speedup.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_speedup", REPO / "scripts" / "plot_speedup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


speedup = load_script()


def summary_for(cells) -> pd.DataFrame:
    """A :func:`plotting.cell_summary` frame built the way the figure gets one -- from per-sample
    rows through the shipped summariser -- so the column shape can never drift from the real one.

    ``cells`` is ``(kernel, framework, milliseconds)``; each cell is given identical samples, which
    keeps the cleaned median exact and the bootstrap CI degenerate (nothing to warn about).
    """
    rows = [dict(benchmark=k, domain="Physics", framework=f, time=t) for k, f, ms in cells for t in [ms] * 5]
    return plotting.cell_summary(pd.DataFrame(rows))


# --- the signed transform ---------------------------------------------------------------------


@pytest.mark.parametrize("ratio,expected", [(1.0, 0.0), (2.0, 1.0), (3.0, 2.0), (0.5, -1.0), (0.25, -3.0)])
def test_the_landmarks_the_spec_names(ratio: float, expected: float) -> None:
    assert speedup.signed_change(ratio) == pytest.approx(expected)


@pytest.mark.parametrize("magnitude", [1.0, 1.25, 1.5, 2.0, 3.0, 10.0, 100.0])
def test_a_win_and_a_loss_of_the_same_size_are_the_same_distance_from_zero(magnitude: float) -> None:
    """⛔ THE point of the figure. On a raw ratio axis a 0.5x regression sits 0.5 below 1.0 while
    the 2.0x win sits 1.0 above it, so the eye reads the regression as the smaller event."""
    win = speedup.signed_change(magnitude)
    loss = speedup.signed_change(1.0 / magnitude)
    assert win == pytest.approx(-loss)
    assert win == pytest.approx(magnitude - 1.0)


@pytest.mark.parametrize("ratio", [0.0, -1.0, -0.5, math.inf, -math.inf, math.nan])
def test_an_unusable_ratio_is_nan_never_zero(ratio: float) -> None:
    """0 means "measured, nothing changed". A cell that was never measured must not claim it."""
    value = speedup.signed_change(ratio)
    assert math.isnan(value), f"{ratio} became {value}, which will be plotted"


# --- band assignment --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,band", [
    (1.0, speedup.BAND_LOW),
    (1.999, speedup.BAND_LOW),
    (0.51, speedup.BAND_LOW),
    (2.0, speedup.BAND_MID),
    (10.0, speedup.BAND_MID),
    (0.5, speedup.BAND_MID),
    (0.1, speedup.BAND_MID),
    (10.5, speedup.BAND_HIGH),
    (100.0, speedup.BAND_HIGH),
    (1.0 / 10.5, speedup.BAND_HIGH),
])
def test_the_band_edges(ratio: float, band: str) -> None:
    """The band named for an edge owns it: 2x and 10x are both ``2x .. 10x``."""
    assert speedup.band_of(speedup.signed_change(ratio)) == band


@pytest.mark.parametrize("magnitude", [1.0, 1.5, 2.0, 9.9, 10.0, 50.0])
def test_a_band_holds_a_win_and_its_mirrored_loss(magnitude: float) -> None:
    """Bands are keyed on magnitude, never on sign -- a 3x regression is read on the same axis as
    a 3x win, which is what makes the panels comparable."""
    assert (speedup.band_of(speedup.signed_change(magnitude)) == speedup.band_of(speedup.signed_change(1.0 /
                                                                                                       magnitude)))


def test_an_unplottable_change_has_no_band() -> None:
    assert speedup.band_of(math.nan) is None


def test_band_limits_are_anchored_at_the_band_edge_and_open_only_at_the_top() -> None:
    """Every panel shows its band's inner edge, so a point's height means the same thing each time
    and a lone ``> 10x`` point is read against the 10x boundary rather than an arbitrary window.
    Only the top band's OUTER end follows the data -- which is why one 100x outlier there cannot
    flatten the panels below it."""
    assert speedup.band_limits(speedup.BAND_LOW, [0.2, -0.3]) == (-1.0, 1.0)
    assert speedup.band_limits(speedup.BAND_MID, [1.5, 4.0]) == (1.0, 9.0)  # wins only
    assert speedup.band_limits(speedup.BAND_MID, [-1.5, -4.0]) == (-9.0, -1.0)  # losses only
    assert speedup.band_limits(speedup.BAND_MID, [-1.5, 4.0]) == (-9.0, 9.0)
    assert speedup.band_limits(speedup.BAND_HIGH, [40.0]) == (9.0, pytest.approx(42.0))
    assert speedup.band_limits(speedup.BAND_HIGH, [-40.0]) == (pytest.approx(-42.0), -9.0)


# --- points from the results summary ------------------------------------------------------------


def test_points_carry_the_median_speedup_over_the_baseline() -> None:
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0)])
    points: List[speedup.Point] = speedup.speedup_points(frame)
    assert len(points) == 1, "the baseline is the divisor, not a series"
    assert points[0].framework == "dace_cpu"
    assert points[0].ratio == pytest.approx(2.0)
    assert points[0].change == pytest.approx(1.0)
    assert points[0].band == speedup.BAND_MID


def test_a_kernel_with_no_baseline_is_dropped_and_named() -> None:
    frame = summary_for([("heat3d", "dace_cpu", 5.0), ("jacobi2d", plotting.BASELINE, 10.0),
                         ("jacobi2d", "dace_cpu", 20.0)])
    with pytest.warns(UserWarning, match="heat3d@dace_cpu"):
        points = speedup.speedup_points(frame)
    assert [point.kernel for point in points] == ["jacobi2d"]
    assert points[0].change == pytest.approx(-1.0), "a 2x slow-down is -1, the mirror of a 2x win"


def test_a_non_positive_median_is_dropped_not_plotted_at_zero() -> None:
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 0.0)])
    with pytest.warns(UserWarning, match="heat3d@dace_cpu"):
        assert speedup.speedup_points(frame) == []


# --- the figures ---------------------------------------------------------------------------------


def rendered_panels(monkeypatch: pytest.MonkeyPatch, points, kernels, output: str) -> int:
    """Render the banded figure and count the panels ON THE FIGURE, not in the code path."""
    seen: List[int] = []
    original = plotting.save_figure

    def spy(path: str, fig) -> str:
        seen.append(len(fig.axes))
        return original(path, fig)

    monkeypatch.setattr(plotting, "save_figure", spy)
    speedup.banded_figure(points, kernels, output)
    assert len(seen) == 1
    return seen[0]


def test_an_empty_band_is_dropped_rather_than_drawn_empty(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: pathlib.Path) -> None:
    """Two kernels, one band -> ONE panel. An empty panel carries no information and its y scale
    would be invented rather than measured, so the band is dropped from the layout."""
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0),
                         ("jacobi2d", plotting.BASELINE, 10.0), ("jacobi2d", "dace_cpu", 2.5)])
    points = speedup.speedup_points(frame)
    assert {point.band for point in points} == {speedup.BAND_MID}
    assert rendered_panels(monkeypatch, points, ["heat3d", "jacobi2d"], str(tmp_path / "speedup.pdf")) == 1


def test_every_non_empty_band_gets_its_own_panel(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Three magnitudes -> three panels, each with its own y scale."""
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 9.5),
                         ("jacobi2d", plotting.BASELINE, 10.0), ("jacobi2d", "dace_cpu", 2.0),
                         ("gemm", plotting.BASELINE, 10.0), ("gemm", "dace_cpu", 0.05)])
    points = speedup.speedup_points(frame)
    assert {point.band for point in points} == {speedup.BAND_LOW, speedup.BAND_MID, speedup.BAND_HIGH}
    assert rendered_panels(monkeypatch, points, ["gemm", "heat3d", "jacobi2d"], str(tmp_path / "speedup.pdf")) == 3


def test_the_simplified_figure_shows_the_band_with_the_most_points(tmp_path: pathlib.Path) -> None:
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0),
                         ("jacobi2d", plotting.BASELINE, 10.0), ("jacobi2d", "dace_cpu", 2.5),
                         ("gemm", plotting.BASELINE, 10.0), ("gemm", "dace_cpu", 9.5)])
    points = speedup.speedup_points(frame)
    assert speedup.dominant_band(points) == speedup.BAND_MID
    out = speedup.simple_figure(points, ["gemm", "heat3d", "jacobi2d"], str(tmp_path / "speedup-simple.svg"))
    blob = pathlib.Path(out).read_bytes()
    assert blob.lstrip().startswith(b"<?xml"), "the simplified variant must be a real SVG"
    assert b"<svg" in blob


def test_every_output_is_written_per_machine(tmp_path: pathlib.Path) -> None:
    """End to end over a real results DB, through the shipped reader: the banded PDF plus the two
    SVG variants, each carrying the machine label (two nodes may never share a figure)."""
    from tests.test_inference_plots import build_results_db

    db = tmp_path / "results.db"
    build_results_db(db, shift=0.5)  # dace_cpu at half the numpy runtime -> a clean 2x
    written = speedup.plot_signed_speedup(db=str(db), preset="S", output=str(tmp_path / "speedup.pdf"), usetex=False)
    pdfs = [p for p in written if p.endswith(".pdf")]
    svgs = sorted(p for p in written if p.endswith(".svg"))
    assert len(pdfs) == 1 and len(svgs) == 2, written
    assert pathlib.Path(pdfs[0]).name.startswith("speedup.")
    assert [pathlib.Path(p).name.split(".")[0] for p in svgs] == ["speedup-mini", "speedup-simple"]
    assert pathlib.Path(pdfs[0]).read_bytes().startswith(b"%PDF-")
    assert all(b"<svg" in pathlib.Path(p).read_bytes() for p in svgs)


def baseline_only_db(path: pathlib.Path) -> None:
    """A results DB carrying the BASELINE and nothing else -- the shipped Result model, so the
    fixture cannot drift from the table the reader queries."""
    from sqlmodel import Session

    from hpcagent_bench.frameworks.schema import Result, results_engine
    with Session(results_engine(str(path))) as session:
        for value in (10.0, 10.5, 9.5):
            session.add(
                Result(timestamp=0,
                       benchmark="heat3d",
                       domain="Physics",
                       preset="S",
                       framework=plotting.BASELINE,
                       agent=None,
                       validated=True,
                       cpu="test-cpu",
                       time=value,
                       native_time=None,
                       datatype="float64",
                       variant=None,
                       prompt_hash=None,
                       execution="native"))
        session.commit()


def test_the_demo_populates_every_band_with_both_signs() -> None:
    """``--demo`` exists so the three panels can be LOOKED at. A change to the synthetic layout
    that emptied a band would quietly turn it back into a one-panel figure, and a demo that only
    ever shows wins would not demonstrate the mirroring it is there to demonstrate."""
    points = speedup.demo_points()
    per_band = {band: {point.kernel for point in points if point.band == band} for band in speedup.BANDS}
    assert all(2 <= len(kernels) <= 3 for kernels in per_band.values()), per_band
    for band in speedup.BANDS:
        assert any(point.change < 0.0 for point in points if point.band == band), f"{band}: no slow-down"
    assert speedup.demo_points() == points, "the demo seed must make the figure reproducible"


def test_a_db_with_only_the_baseline_fails_loudly(tmp_path: pathlib.Path) -> None:
    """No candidate framework means no speed-up exists. Writing no file while exiting 0 is the
    failure that reads as a clean run."""
    db = tmp_path / "baseline_only.db"
    baseline_only_db(db)
    with pytest.warns(UserWarning, match="no kernel has a plottable speed-up"):
        with pytest.raises(RuntimeError, match="no speed-up to plot"):
            speedup.plot_signed_speedup(db=str(db), preset="S", output=str(tmp_path / "speedup.pdf"), usetex=False)
