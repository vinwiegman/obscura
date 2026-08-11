"""Contract tests against the real uniface package.

Everything else in this suite runs offline against fakes, which is fast but
cannot catch uniface renaming a class or changing what ``update`` returns. These
tests skip when uniface is not installed, and are the ones that break when a
version bump moves the ground under the adapters.
"""

from importlib import import_module

import numpy as np
import pytest

from obscura.config import TrackConfig
from obscura.detect import MODELS
from obscura.geometry import Box
from obscura.track import ByteTrackAdapter

pytest.importorskip("uniface", reason="uniface is an optional inference backend")


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_advertised_detector_exists(name):
    """The CLI offers these by name; none may be a typo."""
    module_name, class_name = MODELS[name]
    module = import_module(module_name)
    assert hasattr(module, class_name), f"uniface.{module_name} has no {class_name}"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_detector_accepts_the_kwargs_we_pass(name):
    import inspect

    module_name, class_name = MODELS[name]
    signature = inspect.signature(getattr(import_module(module_name), class_name).__init__)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    for kwarg in ("confidence_threshold", "providers"):
        assert kwarg in signature.parameters or accepts_kwargs


@pytest.fixture
def adapter():
    from uniface.tracking import BYTETracker

    return ByteTrackAdapter(BYTETracker, TrackConfig())


def test_adapter_returns_ids_and_boxes(adapter):
    tracks = adapter.update([(Box(10, 10, 60, 70), 0.9), (Box(200, 30, 250, 90), 0.8)])

    assert len(tracks) == 2
    for track_id, box in tracks:
        assert isinstance(track_id, int)
        assert isinstance(box, Box)
        assert box.width > 0 and box.height > 0


def test_adapter_keeps_ids_stable_across_frames(adapter):
    first = adapter.update([(Box(10, 10, 60, 70), 0.9)])
    second = adapter.update([(Box(14, 10, 64, 70), 0.9)])

    assert first[0][0] == second[0][0]


def test_adapter_survives_a_frame_with_no_detections(adapter):
    """Empty frames still have to reach the tracker: that is how it ages tracks out."""
    adapter.update([(Box(10, 10, 60, 70), 0.9)])

    tracks = adapter.update([])

    assert isinstance(tracks, list)


def test_bytetracker_returns_the_column_layout_we_unpack():
    """We read the track id out of column 4. If that moves, ids become garbage."""
    from uniface.tracking import BYTETracker

    tracker = BYTETracker(track_thresh=0.4, track_buffer=30, match_thresh=0.8)
    detections = np.array([[10, 10, 60, 70, 0.95]], dtype=np.float32)

    output = np.asarray(tracker.update(detections))

    assert output.ndim == 2 and output.shape[1] == 5
    x1, y1, x2, y2, track_id = output[0]
    assert (x1, y1, x2, y2) == pytest.approx((10, 10, 60, 70), abs=1.0)
    assert float(track_id).is_integer()
