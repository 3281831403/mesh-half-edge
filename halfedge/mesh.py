"""Coordinate-free half-edge mesh topology.

Faces are oriented (counter-clockwise) loops of zero-based vertex indices.
A boundary half-edge has ``twin == BOUNDARY`` (``-1``); an interior
half-edge has a reciprocal twin.  ``build()`` returns this schema::

    {
        "faces": [[...], ...],
        "vertices": [int, ...],          # sorted, unique
        "half_edges": [
            {
                "id": int,               # dense index, same as list position
                "origin": int,           # vertex at the half-edge tail
                "face": int,             # owning face index
                "next": int,             # next half-edge around the face
                "twin": int,             # opposite half-edge, or -1
            },
            ...
        ],
    }

Half-edge ids are dense and follow face-corner order: face ``f`` of length
``n`` owns the block ``sum(len(g) for g in faces[:f]) .. +n``.  The ``next``
field closes each face in one cycle.  Every undirected edge is represented
by one half-edge on a boundary or by two reciprocal half-edges in the
interior.  All topology comparisons are integer-exact; floating-point
comparisons use ``TOL`` (absolute tolerance ``1e-9``).  The implementation
must not depend on vertex coordinates.
"""

BOUNDARY = -1
TOL = 1e-9

__all__ = [
    "BOUNDARY",
    "TOL",
    "build",
    "twin_map",
    "vertex_ring",
    "face_ring",
    "flip_edge",
    "euler_characteristic",
    "validate",
    "mesh_batch",
]


def build(faces) -> dict:
    """Build a half-edge mesh from oriented polygon faces.

    ``faces`` is a sequence of sequences of zero-based ``int`` vertex ids.
    Each face must contain at least three corners, must not repeat a vertex,
    and is interpreted as a counter-clockwise loop (for a surface seen from
    outside).  Surfaces may be open; a manifold with boundary and several
    disconnected components are allowed.

    Pairing rules: group all half-edges by their unordered endpoint pair
    ``(min(u, v), max(u, v))``.  An undirected edge with exactly one
    half-edge is a boundary edge and both endpoint references use
    ``twin == -1``.  An undirected edge with exactly two half-edges pairs
    them reciprocally.  An undirected edge with three or more half-edges is
    non-manifold and must raise ``ValueError``; it must never be silently
    overwritten.  If the two half-edges of an edge run in the same direction,
    construction still succeeds (pairing is by undirected edge), and
    ``validate()`` reports the winding error.

    Returns the dict described in the module docstring.  ``vertices`` is the
    sorted unique vertex set, ``faces`` contains fresh lists, and the
    ``half_edges`` list is in face-corner order with dense ``id`` values.
    The input is not mutated and no edge table is shared between calls.

    Raises ``ValueError`` for a face with fewer than three corners, a
    repeated vertex inside a face, a self-loop, a negative vertex id, or an
    undirected edge incident to three or more half-edges.
    """
    raise NotImplementedError("build")


def twin_map(mesh) -> dict:
    """Return the half-edge to twin mapping for *mesh*.

    The result is a new ``dict`` with one item per half-edge:
    ``{half_edge_id: twin_id}``.  Interior half-edges map to their reciprocal
    partner, so ``result[result[h]] == h``.  Boundary half-edges map to
    ``-1``, never to themselves.  The mapping must not invent pairings and
    must not mutate ``mesh``.
    """
    raise NotImplementedError("twin_map")


def vertex_ring(mesh, v) -> list[int]:
    """Return the one-ring neighbours of vertex ``v`` in rotational order.

    Each neighbour vertex appears exactly once.  The ring is complete: it
    includes neighbours reached through ``next`` (edges inside a face) as
    well as neighbours reached through ``twin``, so a boundary vertex loses
    no incident edge.  For an interior vertex the result has length
    ``degree(v)`` and is cyclic in the mesh orientation.  For a boundary
    vertex the walk starts at the outgoing boundary half-edge and ends at
    the other incident boundary edge; the first and last entries are the two
    boundary neighbours and the length is still ``degree(v)``.

    A vertex with no incident edge yields ``[]``.  An unknown vertex raises
    ``KeyError``.  The walk is bounded by the number of half-edges (plus a
    small guard) and must raise ``ValueError`` rather than loop forever on a
    corrupt or open mesh.
    """
    raise NotImplementedError("vertex_ring")


def face_ring(mesh, f) -> list[int]:
    """Return the face ids edge-adjacent to face ``f``.

    Two faces are adjacent only when they share a full undirected edge, not
    merely a vertex.  The result is sorted ascending, contains each adjacent
    face once, and excludes ``f`` itself.  ``f`` is a zero-based face index;
    an unknown face raises ``IndexError``.
    """
    raise NotImplementedError("face_ring")


def flip_edge(mesh, he) -> dict:
    """Flip the interior edge identified by half-edge ``he``.

    Both faces incident to ``he`` must be triangles and must traverse the
    shared edge in opposite directions.  Let the incident faces be
    ``[a, b, c]`` and ``[b, a, d]``, where ``he`` is ``a -> b``.  The flip
    replaces them with ``[c, d, b]`` and ``[d, c, a]``, so the new shared
    edge is ``c -> d``.  The new edge must not already exist.  All peripheral
    ``next``/``twin`` links are rebuilt consistently, including the boundary
    links: the returned mesh must satisfy ``validate(mesh) == []`` whenever
    the input did.

    The input mesh is not mutated.  The return value is a fresh mesh with the
    schema produced by ``build()``; face order is otherwise preserved.  Raise
    ``ValueError`` if ``he`` is out of range, is a boundary half-edge, is not
    part of two triangles, has an inconsistent winding, or would create a
    duplicate edge.
    """
    raise NotImplementedError("flip_edge")


def euler_characteristic(mesh) -> int:
    """Return the Euler characteristic ``V - E + F`` of *mesh*.

    ``V`` is the number of unique vertices, ``F`` the number of faces, and
    ``E`` the number of unique *undirected* edges.  Count an interior edge
    once, not twice: ``E`` is not the half-edge count.  The result is an
    integer; the ``1e-9`` tolerance applies only when the caller compares
    derived floating-point quantities.
    """
    raise NotImplementedError("euler_characteristic")


def validate(mesh) -> list[str]:
    """Check half-edge consistency and return sorted diagnostic strings.

    An empty list means the mesh is structurally valid.  A corrupt mesh is
    diagnosed rather than causing this function to raise.  The checks cover:

    * ``next`` links stay inside their face and form one cycle matching the
      face's corners in order;
    * every twin is in range, is not self, and obeys ``twin(twin) == h``;
    * every interior pair is reciprocal and every boundary half-edge has
      ``twin == -1`` and no opposite mate in the same edge;
    * no undirected edge has more than two incident half-edges;
    * face winding is consistent: the two half-edges of an interior edge
      run in opposite directions (destination checks).  Winding diagnostics
      contain the word ``winding``.

    The returned strings are deterministic and sorted, so callers can use
    ``validate(mesh) == []`` as the success check.
    """
    raise NotImplementedError("validate")


def mesh_batch(faces_list, config) -> dict:
    """Build several meshes in one call, with fully independent edge tables.

    ``faces_list`` is a sequence of face lists accepted by ``build()``.  Each
    mesh is built from scratch; no edge dictionary, half-edge list, or
    mutable state may be reused between meshes (the second mesh must not see
    edge keys from the first, even when vertex ids and edges overlap).

    ``config`` is a dict of optional flags; unknown keys are ignored:
    ``"validate"`` (default ``True``) fills the per-mesh diagnostics and
    ``"euler"`` (default ``True``) fills the per-mesh Euler characteristic.
    A disabled result is ``None`` at that position.

    Returns::

        {
            "meshes": [mesh, ...],            # same order as faces_list
            "validation": [[str, ...] | None, ...],
            "euler": [int | None, ...],
            "config": {...},                   # normalized copy
        }

    The meshes must compare equal to calling ``build()`` individually.
    """
    raise NotImplementedError("mesh_batch")
