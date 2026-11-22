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


def _is_index(value) -> bool:
    """True for a plain ``int`` usable as an index (``bool`` excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _edge_key(u, v):
    """Undirected edge key: endpoints sorted, so both directions collide."""
    return (u, v) if u <= v else (v, u)


def _safe_field(sequence, index, key):
    """Read ``sequence[index][key]`` without raising on malformed input."""
    try:
        item = sequence[index]
    except (IndexError, TypeError):
        return None
    try:
        return item[key]
    except (KeyError, IndexError, TypeError):
        return None


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
    # A fresh edge table per call: nothing is shared between invocations.
    face_list = []
    try:
        face_iter = list(faces)
    except TypeError:
        raise ValueError("faces must be a sequence of sequences") from None

    for f, raw in enumerate(face_iter):
        try:
            corners = list(raw)
        except TypeError:
            raise ValueError(f"face {f}: not a sequence of vertex ids") from None
        if len(corners) < 3:
            raise ValueError(f"face {f}: fewer than three corners")
        for v in corners:
            if not _is_index(v):
                raise ValueError(f"face {f}: vertex id {v!r} is not an int")
            if v < 0:
                raise ValueError(f"face {f}: negative vertex id {v}")
        if len(set(corners)) != len(corners):
            raise ValueError(f"face {f}: repeated vertex (or self-loop)")
        face_list.append(corners)

    half_edges = []
    for f, corners in enumerate(face_list):
        base = len(half_edges)
        n = len(corners)
        for k in range(n):
            half_edges.append(
                {
                    "id": base + k,
                    "origin": corners[k],
                    "face": f,
                    "next": base + (k + 1) % n,
                    "twin": BOUNDARY,
                }
            )

    # Group by *unordered* endpoint pair so both directions share a key.
    edge_map = {}
    for h, he in enumerate(half_edges):
        origin = he["origin"]
        dest = half_edges[he["next"]]["origin"]
        edge_map.setdefault(_edge_key(origin, dest), []).append(h)

    for key, incident in edge_map.items():
        if len(incident) > 2:
            raise ValueError(
                f"non-manifold edge {key}: {len(incident)} incident half-edges"
            )
        if len(incident) == 2:
            first, second = incident
            half_edges[first]["twin"] = second
            half_edges[second]["twin"] = first

    vertices = sorted({v for corners in face_list for v in corners})
    return {
        "faces": [list(corners) for corners in face_list],
        "vertices": vertices,
        "half_edges": half_edges,
    }


def twin_map(mesh) -> dict:
    """Return the half-edge to twin mapping for *mesh*.

    The result is a new ``dict`` with one item per half-edge:
    ``{half_edge_id: twin_id}``.  Interior half-edges map to their reciprocal
    partner, so ``result[result[h]] == h``.  Boundary half-edges map to
    ``-1``, never to themselves.  The mapping must not invent pairings and
    must not mutate ``mesh``.
    """
    result = {}
    for position, he in enumerate(mesh["half_edges"]):
        try:
            half_edge_id = he["id"]
        except (KeyError, TypeError):
            half_edge_id = position
        try:
            twin = he["twin"]
        except (KeyError, TypeError):
            twin = BOUNDARY
        result[half_edge_id] = BOUNDARY if twin == half_edge_id else twin
    return result


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
    if v not in mesh["vertices"]:
        raise KeyError(v)

    half_edges = mesh["half_edges"]
    n = len(half_edges)

    # previous half-edge inside each face, derived from next.
    previous = [BOUNDARY] * n
    for h in range(n):
        nx = _safe_field(half_edges, h, "next")
        if _is_index(nx) and 0 <= nx < n:
            previous[nx] = h

    outgoing = [
        h for h in range(n)
        if _safe_field(half_edges, h, "origin") == v
    ]
    if not outgoing:
        return []

    # Prefer the outgoing boundary half-edge as the start of an open fan.
    start = None
    for h in outgoing:
        if _safe_field(half_edges, h, "twin") == BOUNDARY:
            start = h
            break
    on_boundary = start is not None
    if start is None:
        start = outgoing[0]

    def destination(h):
        nx = _safe_field(half_edges, h, "next")
        if not (_is_index(nx) and 0 <= nx < n):
            raise ValueError(f"half-edge {h}: invalid next link")
        dest = _safe_field(half_edges, nx, "origin")
        if not _is_index(dest):
            raise ValueError(f"half-edge {nx}: invalid origin")
        return dest

    ring = [destination(start)]
    # A boundary vertex has one incident edge that points into it, so the
    # ring is one longer than the number of half-edges leaving the vertex.
    expected_len = len(outgoing) + (1 if on_boundary else 0)
    current = start
    # Bounded walk: a valid ring closes within one traversal of the mesh.
    for _ in range(n + 2):
        prev = previous[current]
        if prev == BOUNDARY:
            raise ValueError(f"vertex {v}: broken prev link")
        twin = _safe_field(half_edges, prev, "twin")
        if twin == BOUNDARY:
            if not on_boundary:
                raise ValueError(f"vertex {v}: open fan around an interior vertex")
            # The incoming boundary edge closes the open fan.
            last = _safe_field(half_edges, prev, "origin")
            if not _is_index(last):
                raise ValueError(f"half-edge {prev}: invalid origin")
            ring.append(last)
            if len(ring) != expected_len:
                raise ValueError(
                    f"vertex {v}: ring covers {len(ring)} of "
                    f"{expected_len} incident edges"
                )
            return ring
        if not (_is_index(twin) and 0 <= twin < n):
            raise ValueError(f"half-edge {prev}: invalid twin link")
        current = twin
        if current == start:
            if len(ring) != expected_len:
                raise ValueError(
                    f"vertex {v}: ring covers {len(ring)} of "
                    f"{expected_len} incident edges"
                )
            return ring
        ring.append(destination(current))
    raise ValueError(f"vertex {v}: ring walk did not terminate")


def face_ring(mesh, f) -> list[int]:
    """Return the face ids edge-adjacent to face ``f``.

    Two faces are adjacent only when they share a full undirected edge, not
    merely a vertex.  The result is sorted ascending, contains each adjacent
    face once, and excludes ``f`` itself.  ``f`` is a zero-based face index;
    an unknown face raises ``IndexError``.
    """
    faces = mesh["faces"]
    face = faces[f]  # IndexError for an unknown face.
    half_edges = mesh["half_edges"]
    n = len(half_edges)

    by_face = {}
    for h in range(n):
        owner = _safe_field(half_edges, h, "face")
        if _is_index(owner):
            by_face.setdefault(owner, []).append(h)

    neighbours = set()
    for h in by_face.get(f, ()):
        twin = _safe_field(half_edges, h, "twin")
        if not _is_index(twin) or twin == BOUNDARY or not (0 <= twin < n):
            continue
        other = _safe_field(half_edges, twin, "face")
        if _is_index(other) and other != f:
            neighbours.add(other)
    return sorted(neighbours)


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
    half_edges = mesh["half_edges"]
    faces = mesh["faces"]
    n = len(half_edges)

    if not _is_index(he) or not (0 <= he < n):
        raise ValueError(f"half-edge {he!r} is out of range")

    twin = _safe_field(half_edges, he, "twin")
    if twin == BOUNDARY:
        raise ValueError(f"half-edge {he} is a boundary half-edge")
    if not _is_index(twin) or not (0 <= twin < n):
        raise ValueError(f"half-edge {he} has an invalid twin {twin!r}")
    if _safe_field(half_edges, twin, "twin") != he:
        raise ValueError(f"half-edge {he} and its twin are not reciprocal")

    face_a = _safe_field(half_edges, he, "face")
    face_b = _safe_field(half_edges, twin, "face")
    if not _is_index(face_a) or not (0 <= face_a < len(faces)):
        raise ValueError(f"half-edge {he} has an invalid face")
    if not _is_index(face_b) or not (0 <= face_b < len(faces)):
        raise ValueError(f"half-edge {twin} has an invalid face")
    if face_a == face_b:
        raise ValueError("both half-edges belong to the same face")

    try:
        tri_a = list(faces[face_a])
        tri_b = list(faces[face_b])
    except TypeError:
        raise ValueError("incident faces are not sequences") from None
    if len(tri_a) != 3 or len(tri_b) != 3:
        raise ValueError("flip_edge requires two incident triangles")

    a = _safe_field(half_edges, he, "origin")
    nx = _safe_field(half_edges, he, "next")
    if not _is_index(a) or not _is_index(nx) or not (0 <= nx < n):
        raise ValueError(f"half-edge {he} has invalid links")
    b = _safe_field(half_edges, nx, "origin")

    twin_origin = _safe_field(half_edges, twin, "origin")
    twin_next = _safe_field(half_edges, twin, "next")
    twin_dest = _safe_field(half_edges, twin_next, "origin") if (
        _is_index(twin_next) and 0 <= twin_next < n
    ) else None
    if twin_origin != b or twin_dest != a:
        raise ValueError("incident triangles have an inconsistent winding")

    if a not in tri_a or b not in tri_a or a not in tri_b or b not in tri_b:
        raise ValueError("face corners disagree with the half-edge links")

    # The third corner of each triangle, i.e. the vertex not on the edge.
    c_candidates = [x for x in tri_a if x != a and x != b]
    d_candidates = [x for x in tri_b if x != a and x != b]
    if len(c_candidates) != 1 or len(d_candidates) != 1:
        raise ValueError("incident triangles do not form a proper flip pair")
    c = c_candidates[0]
    d = d_candidates[0]
    if c in (a, b) or d in (a, b) or c == d:
        raise ValueError("flip would create a degenerate triangle")

    # The new diagonal must not already be an edge of the mesh.
    new_key = _edge_key(c, d)
    for h in range(n):
        origin = _safe_field(half_edges, h, "origin")
        nxt = _safe_field(half_edges, h, "next")
        if not _is_index(origin) or not _is_index(nxt) or not (0 <= nxt < n):
            continue
        dest = _safe_field(half_edges, nxt, "origin")
        if _is_index(dest) and _edge_key(origin, dest) == new_key:
            raise ValueError(f"flip would duplicate existing edge {new_key}")

    new_faces = [list(face) for face in faces]
    new_faces[face_a] = [c, d, b]
    new_faces[face_b] = [d, c, a]
    return build(new_faces)


def euler_characteristic(mesh) -> int:
    """Return the Euler characteristic ``V - E + F`` of *mesh*.

    ``V`` is the number of unique vertices, ``F`` the number of faces, and
    ``E`` the number of unique *undirected* edges.  Count an interior edge
    once, not twice: ``E`` is not the half-edge count.  The result is an
    integer; the ``1e-9`` tolerance applies only when the caller compares
    derived floating-point quantities.
    """
    half_edges = mesh["half_edges"]
    n = len(half_edges)
    vertices = len(set(mesh["vertices"]))
    faces = len(mesh["faces"])

    undirected = set()
    for h in range(n):
        origin = _safe_field(half_edges, h, "origin")
        nxt = _safe_field(half_edges, h, "next")
        if not _is_index(origin) or not _is_index(nxt) or not (0 <= nxt < n):
            continue
        dest = _safe_field(half_edges, nxt, "origin")
        if _is_index(dest):
            undirected.add(_edge_key(origin, dest))
    return vertices - len(undirected) + faces


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
    errors = []
    try:
        raw_faces = mesh["faces"]
        raw_half_edges = mesh["half_edges"]
    except (KeyError, TypeError):
        return ["mesh: missing 'faces' or 'half_edges'"]

    try:
        faces = list(raw_faces)
        half_edges = list(raw_half_edges)
    except TypeError:
        return ["mesh: 'faces' and 'half_edges' must be sequences"]

    n = len(half_edges)
    origins = [_safe_field(half_edges, h, "origin") for h in range(n)]
    nexts = [_safe_field(half_edges, h, "next") for h in range(n)]
    twins = [_safe_field(half_edges, h, "twin") for h in range(n)]
    face_of = [_safe_field(half_edges, h, "face") for h in range(n)]

    corners_by_face = []
    for f, raw in enumerate(faces):
        try:
            corners = list(raw)
        except TypeError:
            errors.append(f"face {f}: not a sequence of corners")
            corners = []
        corners_by_face.append(corners)
        if len(corners) < 3:
            errors.append(f"face {f}: has fewer than three corners")

    for h in range(n):
        owner = face_of[h]
        if not _is_index(owner) or not (0 <= owner < len(faces)):
            errors.append(f"half-edge {h}: face {owner!r} is out of range")

    # --- next links: one cycle per face, following the corner order. ------
    for f, corners in enumerate(corners_by_face):
        owned = [h for h in range(n) if face_of[h] == f]
        if len(owned) != len(corners):
            errors.append(
                f"face {f}: has {len(owned)} half-edges, expected {len(corners)}"
            )

        started = {}
        for h in owned:
            v = origins[h]
            if v not in corners:
                errors.append(
                    f"face {f}: half-edge {h} starts at {v!r}, not a face corner"
                )
            elif v in started:
                errors.append(f"face {f}: vertex {v} starts more than one half-edge")
            else:
                started[v] = h

        for h in owned:
            v = origins[h]
            if v not in corners:
                continue
            k = corners.index(v)
            expected = corners[(k + 1) % len(corners)]
            nxt = nexts[h]
            if not _is_index(nxt) or not (0 <= nxt < n):
                errors.append(f"face {f}: half-edge {h} next {nxt!r} is out of range")
            elif face_of[nxt] != f:
                errors.append(f"face {f}: half-edge {h} next {nxt} leaves the face")
            elif origins[nxt] != expected:
                errors.append(
                    f"face {f}: half-edge {h} next {nxt} breaks the corner cycle"
                )

    # --- undirected edges: manifoldness, boundary pairing, winding. ------
    half_edge_key = [None] * n
    for h in range(n):
        origin = origins[h]
        nxt = nexts[h]
        if not _is_index(origin):
            continue
        if not _is_index(nxt) or not (0 <= nxt < n):
            continue
        dest = origins[nxt]
        if not _is_index(dest):
            continue
        half_edge_key[h] = _edge_key(origin, dest)

    groups = {}
    for h in range(n):
        key = half_edge_key[h]
        if key is not None:
            groups.setdefault(key, []).append(h)

    for key in sorted(groups):
        incident = groups[key]
        if len(incident) > 2:
            errors.append(
                f"edge {key}: non-manifold, {len(incident)} half-edges"
            )
            continue
        if len(incident) == 2:
            first, second = incident
            if twins[first] == BOUNDARY or twins[second] == BOUNDARY:
                errors.append(
                    f"edge {key}: boundary half-edge has an opposite mate"
                )
            if twins[first] != second or twins[second] != first:
                errors.append(f"edge {key}: interior twins are not reciprocal")
            elif origins[first] == origins[second]:
                errors.append(
                    f"edge {key}: winding is inconsistent, both half-edges "
                    f"start at vertex {origins[first]}"
                )
        else:
            only = incident[0]
            if twins[only] != BOUNDARY:
                errors.append(
                    f"edge {key}: boundary half-edge {only} has twin "
                    f"{twins[only]!r} instead of -1"
                )

    # --- twin links. ------------------------------------------------------
    for h in range(n):
        twin = twins[h]
        if twin is None:
            errors.append(f"half-edge {h}: missing twin")
            continue
        if not _is_index(twin):
            errors.append(f"half-edge {h}: twin {twin!r} is not an integer")
            continue
        if twin == BOUNDARY:
            continue
        if not (0 <= twin < n):
            errors.append(f"half-edge {h}: twin {twin} is out of range")
            continue
        if twin == h:
            errors.append(f"half-edge {h}: twin is itself")
            continue
        if twins[twin] != h:
            errors.append(f"half-edge {h}: twin {twin} is not reciprocal")
        key = half_edge_key[h]
        twin_key = half_edge_key[twin]
        if key is not None and twin_key is not None and key != twin_key:
            errors.append(f"half-edge {h}: twin {twin} belongs to a different edge")

    return sorted(set(errors))


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
    normalized = {"validate": True, "euler": True}
    if isinstance(config, dict):
        for key in ("validate", "euler"):
            if key in config:
                normalized[key] = bool(config[key])

    meshes = []
    validation = []
    euler = []
    for faces in faces_list:
        mesh = build(faces)  # fresh edge table for every mesh
        meshes.append(mesh)
        validation.append(validate(mesh) if normalized["validate"] else None)
        euler.append(euler_characteristic(mesh) if normalized["euler"] else None)

    return {
        "meshes": meshes,
        "validation": validation,
        "euler": euler,
        "config": normalized,
    }
