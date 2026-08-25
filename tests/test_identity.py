"""Clustering and the fail-safe rules around it.

The safety argument for this feature lives here: anything that cannot be
identified confidently must end up blurred anyway, and two different people must
never land in one cluster, because deselecting one would un-blur the other.
"""

import numpy as np
import pytest

from obscura.config import IdentityConfig
from obscura.geometry import Box
from obscura.identity import (
    Person,
    TrackSampler,
    _Sample,
    build_people,
    cluster,
    tracks_to_redact,
)
from obscura.types import Detection


def unit(*values) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# --- clustering ----------------------------------------------------------


def test_identical_embeddings_merge():
    embeddings = {1: unit(1, 0, 0), 2: unit(1, 0, 0)}
    assert cluster(embeddings, 0.45) == [[1, 2]]


def test_orthogonal_embeddings_stay_apart():
    embeddings = {1: unit(1, 0, 0), 2: unit(0, 1, 0)}
    assert cluster(embeddings, 0.45) == [[1], [2]]


def test_similar_embeddings_merge_at_a_permissive_threshold():
    embeddings = {1: unit(1, 0, 0), 2: unit(0.9, 0.44, 0)}
    assert cluster(embeddings, 0.85) == [[1, 2]]
    assert cluster(embeddings, 0.95) == [[1], [2]]


def test_average_linkage_refuses_to_chain_two_people_together():
    """A track sitting between two identities must not bridge them.

    Single linkage would merge all three here, which in this tool means
    deselecting one person silently un-blurs another.
    """
    a, bridge, b = unit(1, 0, 0), unit(0.72, 0.69, 0), unit(0, 1, 0)
    groups = cluster({1: a, 2: bridge, 3: b}, 0.7)

    assert [1, 3] not in groups
    assert all(len(group) <= 2 for group in groups)


def test_empty_input_clusters_to_nothing():
    assert cluster({}, 0.45) == []


def test_a_single_track_is_its_own_cluster():
    assert cluster({7: unit(1, 0, 0)}, 0.45) == [[7]]


# --- selection resolution ------------------------------------------------


def people_fixture() -> list[Person]:
    return [
        Person(id=1, track_ids=[10, 11], n_frames=50, first_frame=0, last_frame=60),
        Person(id=2, track_ids=[20], n_frames=30, first_frame=10, last_frame=40),
        Person(
            id=3,
            track_ids=[30],
            n_frames=5,
            first_frame=20,
            last_frame=25,
            always_redact=True,
            reason="face too small to identify",
        ),
    ]


def test_except_policy_blurs_everyone_not_chosen():
    redact = tracks_to_redact(people_fixture(), selected={1}, policy="except")
    assert redact == {20, 30}


def test_only_policy_blurs_just_the_chosen():
    redact = tracks_to_redact(people_fixture(), selected={1}, policy="only")
    assert redact == {10, 11, 30}


def test_unidentifiable_people_are_blurred_under_every_policy():
    """Person 3 is selected for protection, but cannot be identified reliably."""
    assert 30 in tracks_to_redact(people_fixture(), selected={3}, policy="except")
    assert 30 in tracks_to_redact(people_fixture(), selected=set(), policy="only")


def test_selecting_nobody_under_except_blurs_everybody():
    redact = tracks_to_redact(people_fixture(), selected=set(), policy="except")
    assert redact == {10, 11, 20, 30}


def test_all_track_ids_of_a_merged_person_are_redacted_together():
    """Person 1 is two tracks; blurring them must not leave one behind."""
    redact = tracks_to_redact(people_fixture(), selected={2}, policy="except")
    assert {10, 11} <= redact


# --- people construction -------------------------------------------------


def sample(track_id, side=80, embedding=None) -> _Sample:
    return _Sample(
        track_id=track_id,
        frame_index=0,
        area=float(side * side),
        thumbnail=np.zeros((side, side, 3), dtype=np.uint8),
        embedding=embedding,
    )


def test_tracks_with_matching_embeddings_become_one_person():
    spans = {1: (10, 0, 10), 2: (10, 20, 30)}
    samples = {1: sample(1, embedding=unit(1, 0, 0)), 2: sample(2, embedding=unit(1, 0, 0))}

    people = build_people(spans, samples, IdentityConfig())

    assert len(people) == 1
    assert people[0].track_ids == [1, 2]
    assert people[0].n_frames == 20
    assert (people[0].first_frame, people[0].last_frame) == (0, 30)


def test_a_face_too_small_to_identify_is_pinned_to_always_redact():
    spans = {1: (10, 0, 10)}
    samples = {1: sample(1, side=10, embedding=unit(1, 0, 0))}

    person = build_people(spans, samples, IdentityConfig(min_face=40))[0]

    assert person.always_redact
    assert "too small" in person.reason


def test_a_track_without_an_embedding_is_pinned_to_always_redact():
    person = build_people({1: (10, 0, 10)}, {1: sample(1)}, IdentityConfig())[0]

    assert person.always_redact
    assert "landmarks" in person.reason


def test_a_track_with_no_captured_frame_is_pinned_to_always_redact():
    person = build_people({1: (10, 0, 10)}, {}, IdentityConfig())[0]

    assert person.always_redact
    assert person.thumbnail is None


def test_unidentifiable_tracks_are_never_merged_with_each_other():
    """Two unmatched faces are two unknown people, not one."""
    spans = {1: (10, 0, 10), 2: (10, 0, 10)}
    samples = {1: sample(1, side=10), 2: sample(2, side=10)}

    people = build_people(spans, samples, IdentityConfig())

    assert len(people) == 2


def test_people_are_numbered_by_first_appearance():
    spans = {1: (10, 90, 99), 2: (10, 5, 15)}
    samples = {1: sample(1, embedding=unit(1, 0, 0)), 2: sample(2, embedding=unit(0, 1, 0))}

    people = build_people(spans, samples, IdentityConfig())

    assert [p.id for p in people] == [1, 2]
    assert people[0].first_frame == 5


# --- sampling ------------------------------------------------------------


def test_sampler_keeps_the_largest_crop_per_track():
    sampler = TrackSampler(IdentityConfig())
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)

    sampler.observe(frame, 0, 1, Box(10, 10, 40, 40), [])
    sampler.observe(frame, 5, 1, Box(10, 10, 120, 120), [])
    sampler.observe(frame, 9, 1, Box(10, 10, 30, 30), [])

    kept = sampler.samples()[1]
    assert kept.frame_index == 5


def test_sampler_ignores_boxes_outside_the_frame():
    sampler = TrackSampler(IdentityConfig())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    sampler.observe(frame, 0, 1, Box(500, 500, 600, 600), [])

    assert sampler.samples() == {}


def test_sampler_downscales_large_thumbnails():
    sampler = TrackSampler(IdentityConfig(thumb_size=64))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sampler.observe(frame, 0, 1, Box(0, 0, 400, 400), [])

    assert max(sampler.samples()[1].thumbnail.shape[:2]) == 64


@pytest.mark.parametrize("overlap_box", [Box(10, 10, 40, 40), Box(200, 200, 240, 240)])
def test_sampler_survives_detections_that_do_not_match_the_track(overlap_box):
    """Landmarks are recovered by overlap; no match simply means no embedding."""
    sampler = TrackSampler(IdentityConfig())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [Detection(box=overlap_box, score=0.9, landmarks=None)]

    sampler.observe(frame, 0, 1, Box(10, 10, 40, 40), detections)

    assert sampler.samples()[1].aligned is None
