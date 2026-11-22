"""Topology tests for the half-edge mesh exercise.

The suite is intentionally red against the stubs.  Every test calls at least
one public function from ``halfedge.mesh``; none of the tests is skipped or
expected to fail.  The five invariant tests at the bottom cover Euler's
formula, twin involution, vertex adjacency completeness, flip invariance, and
batch independence.
"""

import pytest

from halfedge import (
    BOUNDARY,
    build,
    euler_characteristic,
    face_ring,
    flip_edge,
    mesh_batch,
    twin_map,
    validate,
    vertex_ring,
)

TETRA = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
SQUARE = [[0, 1, 2], [0, 2, 3]]


# ---------------------------------------------------------------------------
# Small, dependency-free accessors used only by the tests.
# ---------------------------------------------------------------------------


def _he(mesh, h):
    return mesh["half_edges"][h]


def _origin(mesh, h):
    return _he(mesh, h)["origin"]


def _next(mesh, h):
    return _he(mesh, h)["next"]


def _twin(mesh, h):
    return _he(mesh, h)["twin"]


def _find_he(mesh, u, v):
    """Return the half-edge id for directed edge u -> v, or None."""
    for he in mesh["half_edges"]:
        if he["origin"] == u and _origin(mesh, he["next"]) == v:
            return he["id"]
    return None


def _undirected_edges(mesh):
    edges = set()
    for he in mesh["half_edges"]:
        a = he["origin"]
        b = _origin(mesh, he["next"])
        edges.add((min(a, b), max(a, b)))
    return edges


def _degree(faces, v):
    neighbours = set()
    for face in faces:
        if v in face:
            neighbours.update(u for u in face if u != v)
    return len(neighbours)


# ---------------------------------------------------------------------------
# One test per hidden defect (10).
# ---------------------------------------------------------------------------


def test_build_keeps_both_directions_of_shared_edge():
    mesh = build(SQUARE)
    h = _find_he(mesh, 0, 2)
    assert h is not None
    t = _twin(mesh, h)
    assert t != BOUNDARY
    assert _twin(mesh, t) == h
    for he in mesh["half_edges"]:
        if he["twin"] != BOUNDARY:
            assert _twin(mesh, he["twin"]) == he["id"]


def test_build_rejects_non_manifold_edge():
    with pytest.raises(ValueError):
        build([[0, 1, 2], [1, 0, 3], [0, 1, 4]])


def test_boundary_twin_is_minus_one_and_ring_terminates():
    mesh = build([[0, 1, 2]])
    tm = twin_map(mesh)
    assert set(tm) == {0, 1, 2}
    assert all(t == BOUNDARY for t in tm.values())
    ring = vertex_ring(mesh, 0)
    assert sorted(ring) == [1, 2]


def test_vertex_ring_walks_next_not_only_twin():
    mesh = build(SQUARE)
    ring = vertex_ring(mesh, 0)
    assert len(ring) == 3
    assert set(ring) == {1, 2, 3}
    for a, b in zip(ring, ring[1:]):
        assert any(
            0 in face and a in face and b in face for face in mesh["faces"]
        )


def test_vertex_ring_open_boundary_has_no_lost_neighbour():
    mesh = build([[0, 1, 2]])
    ring = vertex_ring(mesh, 1)
    assert len(ring) == 2
    assert set(ring) == {0, 2}
    assert len(set(ring)) == len(ring)


def test_face_ring_requires_shared_edge():
    vertex_only = build([[0, 1, 2], [0, 3, 4]])
    assert face_ring(vertex_only, 0) == []

    edge_shared = build([[0, 1, 2], [2, 1, 3]])
    assert face_ring(edge_shared, 0) == [1]
    assert face_ring(edge_shared, 1) == [0]


def test_flip_edge_repairs_peripheral_twins():
    mesh = build(SQUARE)
    h = _find_he(mesh, 0, 2)
    assert h is not None
    flipped = flip_edge(mesh, h)

    assert validate(flipped) == []
    assert mesh["faces"] == SQUARE
    for he in flipped["half_edges"]:
        t = he["twin"]
        if t == BOUNDARY:
            continue
        assert flipped["half_edges"][t]["twin"] == he["id"]
    assert _find_he(flipped, 1, 3) is not None
    assert _find_he(flipped, 0, 2) is None


def test_euler_characteristic_uses_undirected_edges():
    mesh = build(SQUARE)
    chi = euler_characteristic(mesh)
    assert len(mesh["half_edges"]) == 6
    assert abs(chi - (4 - 5 + 2)) < 1e-9
    assert chi == 1


def test_validate_reports_inconsistent_winding():
    assert validate(build(TETRA)) == []
    # One face is reversed while the other three keep the outward winding.
    bad = build([[0, 2, 1], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    errors = validate(bad)
    assert errors
    assert any(
        "winding" in e.lower() or "orientation" in e.lower() or "绕序" in e
        for e in errors
    )


def test_mesh_batch_does_not_share_edge_tables():
    first = [[0, 1, 2], [0, 2, 3]]
    second = [[0, 1, 2], [0, 2, 4]]
    batch = mesh_batch([first, second], {})

    assert len(batch["meshes"]) == 2
    assert batch["meshes"][0] == build(first)
    assert batch["meshes"][1] == build(second)
    assert batch["validation"] == [[], []]
    assert batch["euler"] == [1, 1]


# ---------------------------------------------------------------------------
# Five topology invariants.
# ---------------------------------------------------------------------------


def test_tetrahedron_euler_formula():
    mesh = build(TETRA)
    assert len(mesh["vertices"]) == 4
    assert len(mesh["faces"]) == 4
    assert len(_undirected_edges(mesh)) == 6
    chi = euler_characteristic(mesh)
    assert abs(chi - 2) < 1e-9


def test_twin_mapping_is_an_involution():
    for faces in (TETRA, [[0, 1, 2]]):
        mesh = build(faces)
        tm = twin_map(mesh)
        for h, t in tm.items():
            if t == BOUNDARY:
                assert _twin(mesh, h) == BOUNDARY
            else:
                assert t != h
                assert tm[t] == h


def test_vertex_ring_size_equals_degree():
    mesh = build(SQUARE)
    for v in mesh["vertices"]:
        ring = vertex_ring(mesh, v)
        assert len(ring) == _degree(SQUARE, v)
        assert len(set(ring)) == len(ring)


def test_flip_edge_preserves_euler_characteristic():
    mesh = build(SQUARE)
    h = _find_he(mesh, 0, 2)
    assert h is not None
    before = euler_characteristic(mesh)
    after = euler_characteristic(flip_edge(mesh, h))
    assert abs(before - after) < 1e-9
    assert before == 1


def test_batch_matches_independent_builds():
    specs = [TETRA, [[0, 1, 2]]]
    batch = mesh_batch(specs, {"validate": True, "euler": True})

    assert len(batch["meshes"]) == len(specs)
    for i, spec in enumerate(specs):
        single = build(spec)
        assert batch["meshes"][i] == single
        assert batch["validation"][i] == validate(single)
        assert batch["euler"][i] == euler_characteristic(single)
    assert batch["euler"] == [2, 1]
    assert batch["validation"] == [[], []]
