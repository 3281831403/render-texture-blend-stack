"""Multi-layer material texture blend stack.

This module is the single place where a material's layer stack is ordered,
composited, merged and validated. Everything here is pure: no I/O, no
randomness, no third-party packages (the test suite only needs the standard
library and pytest).

Shared conventions
==================

* A **color** is a 3-tuple ``(r, g, b)`` of floats in the closed range
  ``0.0..1.0``.
* A **normal** is a 3-tuple ``(x, y, z)`` of floats in ``-1.0..1.0``. It is
  treated as a direction: it is normalized before any math, and every blended
  normal is returned with length ``1.0``.
* ``mask`` and ``opacity`` are floats in ``0.0..1.0``. ``mask`` is the coverage
  of a layer inside its own footprint, ``opacity`` is the material opacity.
  Their product, ``eff = opacity * mask``, is the layer's effective alpha and
  is applied exactly once.
* A **layer** is a dict with the keys ``"id"`` (str, unique inside one stack),
  ``"color"`` (color tuple), ``"mask"`` (float), ``"opacity"`` (float) and
  ``"normal"`` (normal tuple). The key ``"order"`` (int or float) is optional.
* Stack order is painter's order: the first layer is the bottom of the stack,
  the last layer is the top. ``order`` sorts ascending, so the smallest
  ``order`` is painted first (bottom).
* Float comparisons in the test suite use a tolerance of ``1e-6``; no function
  is expected to round its results.
"""

import copy
import math

TOLERANCE = 1e-6


class StackError(ValueError):
    """Raised when a layer stack, a layer or a normal cannot be used."""


def _clamp01(value):
    """Clamp a scalar channel into the closed range ``0.0..1.0``."""
    return max(0.0, min(1.0, value))


def _in_range(value):
    """True when ``value`` is a real number inside ``0.0..1.0`` (inclusive)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and 0.0 <= float(value) <= 1.0


def _normalize_vector(vector):
    """Return ``vector`` scaled to unit length.

    A zero-length vector cannot define a direction and raises
    :class:`StackError`.
    """
    length = math.sqrt(sum(component * component for component in vector))
    if length == 0.0:
        raise StackError("zero-length normal cannot be normalized")
    return tuple(component / length for component in vector)


def normalize_layers(layers):
    """Return a new list of ``layers`` sorted by ``order``, ascending.

    Contract:

    * every layer that has a numeric ``"order"`` sorts before every layer that
      has none;
    * layers without ``"order"`` are appended at the end of the result and keep
      their relative input order;
    * the sort is stable, so layers with an equal ``order`` keep their relative
      input order;
    * the input list and the layer dicts are not modified -- a new list is
      returned, but its items are the original dicts;
    * an empty input returns an empty list.

    Sorting by insertion order, or ignoring ``"order"`` entirely, is a bug: the
    bottom layer must be the one with the smallest ``order`` whatever its
    position in the input list.
    """
    ordered = [layer for layer in layers if "order" in layer]
    unordered = [layer for layer in layers if "order" not in layer]
    # Python's sort is stable, so equal orders keep their relative input order.
    ordered.sort(key=lambda layer: layer["order"])
    return ordered + unordered


def blend_over(base, layer):
    """Composite ``layer`` over ``base`` and return the resulting color tuple.

    ``base`` is a color tuple, or ``None`` meaning "transparent black, nothing
    below the layer". The layer's effective alpha is computed exactly once::

        alpha = layer["opacity"] * layer["mask"]

    (``opacity`` defaults to ``1.0`` and ``mask`` to ``1.0`` when the keys are
    absent). The result is the standard premultiplied *over*::

        out = layer["color"] * alpha + base * (1.0 - alpha)   # per channel
        out = layer["color"] * alpha                          # base is None

    Consequences pinned down by the contract:

    * an opaque layer (``alpha == 1.0``) over ``None`` is exactly the layer
      color;
    * a fully transparent layer (``alpha == 0.0``) leaves ``base`` untouched;
    * a layer at ``alpha == 0.5`` contributes exactly half of its own color and
      half of the base -- the alpha must never be multiplied in twice, which
      would darken the edges of every semi-transparent layer.

    Channels are clamped to ``0.0..1.0``.
    """
    alpha = layer.get("opacity", 1.0) * layer.get("mask", 1.0)
    layer_color = layer["color"]

    if base is None:
        return tuple(
            _clamp01(channel * alpha) for channel in layer_color
        )

    return tuple(
        _clamp01(
            layer_color[index] * alpha
            + base[index] * (1.0 - alpha)
        )
        for index in range(3)
    )


def blend_normal_map(base_n, layer_n, mask):
    """Blend the detail normal ``layer_n`` into ``base_n`` (reoriented).

    Both inputs are normalized first, then ``layer_n`` is reoriented into the
    tangent frame of ``base_n`` with the classic reoriented normal mapping step
    (all in ``-1.0..1.0`` space)::

        t = base + (0.0, 0.0, 1.0)
        u = layer * (-1.0, -1.0, 1.0)
        rnm = normalize(t * dot(t, u) - u * t.z)

    Consequences the contract pins down:

    * a flat base normal ``(0.0, 0.0, 1.0)`` reproduces ``layer_n`` exactly;
    * a flat detail normal ``(0.0, 0.0, 1.0)`` leaves ``base_n`` unchanged;
    * the result is always a unit vector (length ``1.0`` within ``1e-6``); it is
      never the raw, unnormalized sum of two normals, which would push
      ``z`` above ``1.0`` and brighten the lighting everywhere.

    ``mask`` in ``0.0..1.0`` is the blend weight: ``0.0`` returns the
    normalized base normal, ``1.0`` returns the full reoriented blend, and
    intermediate values interpolate linearly between the two before the final
    normalization. A zero-length input normal, or a mask outside
    ``0.0..1.0``, raises :class:`StackError`.
    """
    if not _in_range(mask):
        raise StackError(f"mask {mask!r} is outside the range 0.0..1.0")

    base = _normalize_vector(base_n)
    layer = _normalize_vector(layer_n)

    t = (base[0], base[1], base[2] + 1.0)
    u = (-layer[0], -layer[1], layer[2])
    dot_tu = t[0] * u[0] + t[1] * u[1] + t[2] * u[2]

    reoriented = (
        t[0] * dot_tu - u[0] * t[2],
        t[1] * dot_tu - u[1] * t[2],
        t[2] * dot_tu - u[2] * t[2],
    )
    reoriented_length = math.sqrt(
        sum(component * component for component in reoriented)
    )
    if reoriented_length == 0.0:
        # Degenerate geometry (e.g. a base normal pointing straight down):
        # there is no tangent frame to reorient into, keep the base direction.
        rnm = base
    else:
        rnm = tuple(component / reoriented_length for component in reoriented)

    blended = tuple(
        base[index] * (1.0 - mask) + rnm[index] * mask
        for index in range(3)
    )
    return _normalize_vector(blended)


def apply_mask(color, mask, invert):
    """Attenuate ``color`` by the effective mask coverage.

    The effective coverage is ``mask`` when ``invert`` is false and
    ``1.0 - mask`` when ``invert`` is true, so inverting a mask of ``m`` always
    yields the exact complement ``1.0 - m``. Every channel is multiplied by that
    coverage::

        out[i] = color[i] * coverage

    A channel that is fully covered (coverage ``1.0``) is returned unchanged; a
    channel that is fully masked out (coverage ``0.0``) becomes ``0.0``.
    Ignoring ``invert`` is a bug: for ``mask == 0.25`` the forward result and
    the inverted result must differ. A ``mask`` outside ``0.0..1.0`` raises
    :class:`StackError`.
    """
    if not _in_range(mask):
        raise StackError(f"mask {mask!r} is outside the range 0.0..1.0")

    coverage = 1.0 - mask if invert else mask
    return tuple(_clamp01(channel * coverage) for channel in color)


def _composite(layers, validate):
    """Internal compositing entry point used by :func:`resolve` and batches."""
    if not layers:
        raise StackError("empty stack: a stack with no layers has no result")

    if validate:
        problems = validate_stack(layers)
        if problems:
            raise StackError("; ".join(problems))

    ordered = normalize_layers(layers)

    color = None                     # transparent black
    normal = (0.0, 0.0, 1.0)         # flat
    for layer in ordered:
        # The effective alpha is computed once and feeds both the color and
        # the normal blend weight, so mask and opacity each take effect once.
        eff = layer.get("opacity", 1.0) * layer.get("mask", 1.0)
        color = blend_over(color, layer)
        normal = blend_normal_map(normal, layer["normal"], eff)

    return {"color": tuple(color), "normal": tuple(normal)}


def resolve(layers):
    """Composite a whole stack; return ``{"color": ..., "normal": ...}``.

    Ordering and validation:

    * the layers are ordered with :func:`normalize_layers` before use;
    * ``layers == []`` raises :class:`StackError` -- an empty stack has no
      defined color or normal;
    * otherwise :func:`validate_stack` is run first and any reported problem
      raises a single :class:`StackError` whose message contains every problem,
      so an invalid stack (bad ``opacity``, duplicate ``id``, ...) never
      produces a color;
    * the layer dicts are never modified.

    Compositing, bottom to top::

        color  = None                        # transparent black
        normal = (0.0, 0.0, 1.0)             # flat
        for layer in ordered:
            eff = layer["opacity"] * layer["mask"]     # computed once
            color  = blend_over(color, layer)          # uses the same eff
            normal = blend_normal_map(normal, layer["normal"], eff)

    ``mask`` and ``opacity`` therefore each take effect exactly once: the
    layer color is never pre-masked and then alpha-blended with the same mask
    again (that would make every masked layer far too transparent). A layer
    with ``eff == 0.0`` is transparent and changes neither color nor normal.

    The returned color channels are clamped to ``0.0..1.0`` and the returned
    normal is a unit vector.
    """
    return _composite(layers, validate=True)


def merge_layers(a, b):
    """Merge the adjacent layers ``a`` (lower) and ``b`` (upper) into one dict.

    The merged layer covers only what both layers cover -- the masks are
    multiplied (intersection), never unioned or maxed, so a merge can never make
    the layer cover more than before::

        {
            "id":      a["id"] + "+" + b["id"],
            "color":   blend_over(a["color"], b),
            "mask":    a["mask"] * b["mask"],
            "opacity": a["opacity"] * b["opacity"],
            "normal":  blend_normal_map(a["normal"], b["normal"],
                                        a["mask"] * b["mask"]),
            "order":   the smaller ``order`` of the two inputs,
        }

    ``"order"`` is omitted when neither input has one. Missing ``mask`` and
    ``opacity`` default to ``1.0``; a missing ``normal`` defaults to
    ``(0.0, 0.0, 1.0)``. The two input dicts are not modified. A zero-length
    normal in either input raises :class:`StackError`.
    """
    a_mask = a.get("mask", 1.0)
    b_mask = b.get("mask", 1.0)
    a_opacity = a.get("opacity", 1.0)
    b_opacity = b.get("opacity", 1.0)
    a_normal = a.get("normal", (0.0, 0.0, 1.0))
    b_normal = b.get("normal", (0.0, 0.0, 1.0))

    merged_mask = a_mask * b_mask

    merged = {
        "id": a["id"] + "+" + b["id"],
        "color": blend_over(a["color"], b),
        "mask": merged_mask,
        "opacity": a_opacity * b_opacity,
        "normal": blend_normal_map(a_normal, b_normal, merged_mask),
    }

    orders = [
        layer["order"] for layer in (a, b) if "order" in layer
    ]
    if orders:
        merged["order"] = min(orders)

    return merged


def validate_stack(layers):
    """Return a list of problems; ``[]`` when the stack is usable.

    Every check reports one string per offending layer, each containing a
    lowercase keyword:

    * ``"empty stack"`` -- ``layers`` is empty (nothing to composite);
    * ``"duplicate id: <id>"`` -- two layers share the same ``"id"``;
    * ``"opacity out of range: <id>"`` -- ``opacity`` is outside ``0.0..1.0``;
    * ``"mask out of range: <id>"`` -- ``mask`` is outside ``0.0..1.0``;
    * ``"zero normal: <id>"`` -- ``normal`` has length ``0.0``.

    The function never raises for ordinary malformed data and never modifies
    the layers. An out-of-range ``opacity`` must be reported instead of being
    silently composited, which would push color channels above ``1.0``.
    """
    problems = []

    if not layers:
        problems.append("empty stack: no layers to composite")
        return problems

    seen_ids = set()
    for layer in layers:
        layer_id = layer.get("id", "<missing id>")
        if layer_id in seen_ids:
            problems.append(f"duplicate id: {layer_id}")
        seen_ids.add(layer_id)

    for layer in layers:
        layer_id = layer.get("id", "<missing id>")

        if not _in_range(layer.get("opacity", 1.0)):
            problems.append(f"opacity out of range: {layer_id}")

        if not _in_range(layer.get("mask", 1.0)):
            problems.append(f"mask out of range: {layer_id}")

        normal = layer.get("normal", (0.0, 0.0, 1.0))
        normal_length = math.sqrt(
            sum(component * component for component in normal)
        )
        if normal_length == 0.0:
            problems.append(f"zero normal: {layer_id}")

    return problems


def stack_batch(materials, config):
    """Composite many materials in one call, atomically.

    ``materials`` is a list of ``{"id": str, "layers": [layer, ...]}``.
    ``config`` is a dict; the recognised keys are ``"strict"`` (default
    ``True``) and ``"validate"`` (default ``True``, forwarded to
    :func:`resolve`).

    Returns ``{"results": [...], "errors": [...]}`` where every result is
    ``{"id": <material id>, "color": tuple, "normal": tuple}``, ordered exactly
    like the input ``materials``: a successful batch never reorders them.

    * **Isolation** -- every material is composited from its own deep copy of
      its layers, so no layer list and no layer dict is shared between two
      materials, and the caller's ``materials`` argument is never mutated.
    * **Rollback** -- with ``strict`` true, if *any* material is invalid
      (missing keys, empty stack, :func:`validate_stack` problems) the whole
      batch is rejected with :class:`StackError`; no partial ``results`` are
      returned and the previously composited materials leave no state behind.
    * With ``strict`` false the failing material is skipped and recorded in
      ``"errors"`` as ``{"id": ..., "problems": [...]}``, while the valid
      materials keep their input order in ``"results"``.
    * ``materials == []`` returns ``{"results": [], "errors": []}``; an empty
      batch is not an error.

    Layers are assumed to be complete dicts (keys ``id``, ``color``, ``mask``,
    ``opacity``, ``normal``, optional ``order``); mutable state is copied,
    never aliased.
    """
    strict = config.get("strict", True)
    do_validate = config.get("validate", True)

    results = []
    errors = []

    for material in materials:
        material_id = material.get("id", "<missing id>")
        problems = []
        try:
            # A deep copy per material guarantees that neither the layer list
            # nor any layer dict is shared across materials or with the caller.
            layers_copy = copy.deepcopy(material["layers"])

            if do_validate:
                problems = validate_stack(layers_copy)
                if problems:
                    raise StackError("; ".join(problems))

            composite = _composite(layers_copy, validate=False)
        except StackError as error:
            if strict:
                raise StackError(
                    f"material {material_id!r} failed: {error}"
                ) from error
            errors.append(
                {"id": material_id, "problems": problems or [str(error)]}
            )
            continue
        except (KeyError, TypeError) as error:
            # Missing required keys / malformed structures are data errors too.
            if strict:
                raise StackError(
                    f"material {material_id!r} failed: {error}"
                ) from error
            errors.append({"id": material_id, "problems": [str(error)]})
            continue

        results.append(
            {
                "id": material_id,
                "color": composite["color"],
                "normal": composite["normal"],
            }
        )

    return {"results": results, "errors": errors}
