"""Specification tests for the material blend stack.

Every test fails on the initial ``NotImplementedError`` stubs. The file is
read-only for the person fixing the task: make these green by implementing
``matstack/stack.py``, never by editing ``tests/test_stack.py``.
"""

import math

import pytest

from matstack.stack import (
    StackError,
    apply_mask,
    blend_normal_map,
    blend_over,
    merge_layers,
    normalize_layers,
    resolve,
    stack_batch,
    validate_stack,
)

TOL = 1e-6


# --- helpers -----------------------------------------------------------------


def make_layer(
    layer_id,
    color=(0.5, 0.5, 0.5),
    mask=1.0,
    opacity=1.0,
    normal=(0.0, 0.0, 1.0),
    order=None,
):
    """Build a layer dict with the five required keys and an optional order."""
    layer = {
        "id": layer_id,
        "color": color,
        "mask": mask,
        "opacity": opacity,
        "normal": normal,
    }
    if order is not None:
        layer["order"] = order
    return layer


def length(vector):
    return math.sqrt(sum(component * component for component in vector))


def assert_rgb(actual, expected, label="color"):
    assert len(actual) == 3, f"{label}: expected 3 channels, got {actual!r}"
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert abs(got - want) <= TOL, (
            f"{label}: channel {index} is {got!r}, expected {want!r} "
            f"(tolerance {TOL})"
        )


def assert_normal(actual, expected, label="normal"):
    assert len(actual) == 3, f"{label}: expected 3 components, got {actual!r}"
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert abs(got - want) <= TOL, (
            f"{label}: component {index} is {got!r}, expected {want!r} "
            f"(tolerance {TOL})"
        )


# --- one test per seeded defect ---------------------------------------------


def test_layer_order_sorts_by_order_key():
    """Defect 1: layers must be sorted by ``order``, not by insertion order."""
    layers = [
        make_layer("top", color=(1.0, 0.0, 0.0), order=30),
        make_layer("middle", color=(0.0, 1.0, 0.0), order=10),
        make_layer("loose", color=(0.0, 0.0, 1.0)),  # no order -> appended last
        make_layer("bottom", color=(0.5, 0.5, 0.5), order=20),
    ]

    ordered_ids = [layer["id"] for layer in normalize_layers(layers)]

    assert ordered_ids == ["middle", "bottom", "top", "loose"]
    # the input list itself must be left alone
    assert [layer["id"] for layer in layers] == ["top", "middle", "loose", "bottom"]


def test_over_applies_alpha_exactly_once():
    """Defect 2: premultiplied over must not multiply alpha in twice."""
    base = (0.2, 0.4, 0.6)
    glass = make_layer("glass", color=(1.0, 0.5, 0.0), opacity=0.5, mask=1.0)

    got = blend_over(base, glass)

    alpha = 0.5
    want = (
        alpha * 1.0 + (1.0 - alpha) * 0.2,
        alpha * 0.5 + (1.0 - alpha) * 0.4,
        alpha * 0.0 + (1.0 - alpha) * 0.6,
    )
    assert_rgb(got, want)


def test_normal_blend_is_not_an_unnormalized_sum():
    """Defect 3: additive normals must be renormalized, never over-bright."""
    base_n = (math.sqrt(0.5), 0.0, math.sqrt(0.5))
    detail_n = (0.0, math.sqrt(0.5), math.sqrt(0.5))

    got = blend_normal_map(base_n, detail_n, 1.0)

    assert length(got) <= 1.0 + TOL, f"over-bright normal: {got!r}"
    assert got[2] <= 1.0 + TOL, f"z above 1.0 means the sum was not normalized: {got!r}"
    assert got[2] < 1.0 - 1e-3, f"additive blend leaks through: {got!r}"


def test_normal_blend_uses_layer_detail():
    """Defect 4: the detail normal must be reoriented in, not discarded."""
    flat = (0.0, 0.0, 1.0)
    detail = (0.6, 0.0, 0.8)  # unit length

    from_flat_base = blend_normal_map(flat, detail, 1.0)
    assert_normal(from_flat_base, detail)
    assert abs(from_flat_base[0]) > 0.1, "detail tilt was dropped"

    # reorientation: a flat detail must leave a tilted base normal untouched
    assert_normal(blend_normal_map(detail, flat, 1.0), detail)


def test_apply_mask_honours_invert():
    """Defect 5: ``invert`` must flip the coverage, not be ignored."""
    color = (0.8, 0.4, 0.2)

    forward = apply_mask(color, 0.25, False)
    inverted = apply_mask(color, 0.25, True)

    assert_rgb(forward, (0.2, 0.1, 0.05), label="forward mask")
    assert_rgb(inverted, (0.6, 0.3, 0.15), label="inverted mask")


def test_mask_and_opacity_are_applied_once():
    """Defect 6: mask * opacity is the alpha and must not be applied twice."""
    bottom = make_layer("base", color=(0.0, 0.0, 0.0), mask=1.0, opacity=1.0)
    film = make_layer("film", color=(1.0, 1.0, 1.0), mask=0.5, opacity=0.5)

    composite = resolve([bottom, film])

    alpha = 0.5 * 0.5
    assert_rgb(composite["color"], (alpha, alpha, alpha))


def test_merge_layers_multiplies_masks():
    """Defect 7: merged masks intersect (product), they never union."""
    lower = make_layer(
        "base", color=(0.2, 0.2, 0.2), mask=0.5, opacity=0.5, order=0
    )
    upper = make_layer(
        "detail", color=(1.0, 0.0, 0.0), mask=0.4, opacity=0.3, order=1
    )

    merged = merge_layers(lower, upper)

    assert abs(merged["mask"] - 0.2) <= TOL, (
        f"merged mask is {merged['mask']!r}, expected the product 0.2"
    )
    assert abs(merged["opacity"] - 0.15) <= TOL
    assert merged["id"] == "base+detail"
    assert_normal(merged["normal"], (0.0, 0.0, 1.0))

    alpha = 0.3 * 0.4  # upper layer's effective alpha
    want_color = (
        alpha * 1.0 + (1.0 - alpha) * 0.2,
        alpha * 0.0 + (1.0 - alpha) * 0.2,
        alpha * 0.0 + (1.0 - alpha) * 0.2,
    )
    assert_rgb(merged["color"], want_color, label="merged color")


def test_validate_rejects_opacity_out_of_range():
    """Defect 8: an out-of-range opacity is a problem, not extra brightness."""
    stack = [make_layer("base"), make_layer("glow", opacity=1.5)]

    problems = validate_stack(stack)

    assert problems, "validate_stack accepted opacity=1.5"
    assert any("opacity" in problem.lower() for problem in problems), problems
    with pytest.raises(StackError):
        resolve([make_layer("glow", opacity=1.5)])


def test_stack_batch_isolates_materials_and_rolls_back():
    """Defect 9: materials must not share layer state, and batches roll back."""
    materials = [
        {
            "id": "steel",
            "layers": [
                make_layer("steel_base", color=(0.0, 0.0, 0.0)),
                make_layer("steel_top", color=(1.0, 1.0, 1.0)),
            ],
        },
        {
            "id": "wood",
            "layers": [make_layer("wood_base", color=(0.6, 0.4, 0.2))],
        },
    ]

    batch = stack_batch(materials, {"strict": True})

    assert [result["id"] for result in batch["results"]] == ["steel", "wood"]
    assert_rgb(batch["results"][0]["color"], (1.0, 1.0, 1.0), label="steel")
    assert_rgb(batch["results"][1]["color"], (0.6, 0.4, 0.2), label="wood")
    assert len(materials[0]["layers"]) == 2, "batch mutated the steel layer list"
    assert len(materials[1]["layers"]) == 1, "batch mutated the wood layer list"

    rejected = [
        {"id": "ok", "layers": [make_layer("ok_base", color=(0.1, 0.1, 0.1))]},
        {
            "id": "bad",
            "layers": [make_layer("bad_base", color=(0.9, 0.9, 0.9), opacity=2.0)],
        },
    ]
    with pytest.raises(StackError):
        stack_batch(rejected, {"strict": True})
    assert len(rejected[0]["layers"]) == 1, "a failed batch left partial state behind"


def test_validate_flags_empty_stack():
    """Defect 10: an empty stack is a reported problem, not a silent value."""
    problems = validate_stack([])

    assert problems, "validate_stack([]) reported no problem"
    assert any("empty" in problem.lower() for problem in problems), problems
    assert validate_stack([make_layer("only")]) == []


# --- combination tests -------------------------------------------------------


def test_single_layer_over_empty_bottom_is_layer_color():
    """One opaque layer over ``None`` must reproduce its own color."""
    only = make_layer("only", color=(0.25, 0.5, 0.75), mask=1.0, opacity=1.0)

    assert_rgb(blend_over(None, only), (0.25, 0.5, 0.75), label="blend_over(None)")
    assert_rgb(resolve([only])["color"], (0.25, 0.5, 0.75), label="resolve")


def test_normal_blend_result_has_unit_length():
    """Every blended normal must be a unit vector within 1e-6."""
    pairs = [
        ((0.0, 0.0, 1.0), (0.6, 0.0, 0.8), 1.0),
        ((0.6, 0.0, 0.8), (0.0, 0.0, 1.0), 1.0),
        ((0.48, 0.0, 0.877), (0.0, 0.48, 0.877), 1.0),
        ((0.6, 0.0, 0.8), (0.0, 0.6, 0.8), 0.5),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.0),
    ]

    for base_n, detail_n, weight in pairs:
        got = blend_normal_map(base_n, detail_n, weight)
        assert abs(length(got) - 1.0) <= TOL, (
            f"blend_normal_map({base_n!r}, {detail_n!r}, {weight!r}) "
            f"returned a vector of length {length(got)!r}"
        )


def test_empty_stack_raises_stack_error():
    """An empty stack has no defined result and must raise ``StackError``."""
    with pytest.raises(StackError):
        resolve([])


def test_zero_opacity_layer_does_not_change_result():
    """A layer with ``opacity == 0`` must not touch color or normal."""
    paint = make_layer("paint", color=(0.2, 0.6, 0.4), mask=1.0, opacity=1.0, order=0)
    ghost = make_layer(
        "ghost",
        color=(1.0, 0.0, 0.0),
        mask=1.0,
        opacity=0.0,
        normal=(0.6, 0.0, 0.8),
        order=1,
    )

    with_ghost = resolve([paint, ghost])
    without_ghost = resolve([paint])

    assert_rgb(with_ghost["color"], without_ghost["color"], label="ghost color")
    assert_normal(with_ghost["normal"], without_ghost["normal"], label="ghost normal")


def test_stack_batch_preserves_material_order_on_success():
    """A successful batch returns results in the input material order."""
    materials = [
        {"id": "brick", "layers": [make_layer("brick_base", color=(0.5, 0.2, 0.2))]},
        {
            "id": "asphalt",
            "layers": [make_layer("asphalt_base", color=(0.1, 0.1, 0.1))],
        },
        {
            "id": "glass",
            "layers": [make_layer("glass_base", color=(0.7, 0.9, 1.0))],
        },
    ]

    batch = stack_batch(materials, {"strict": True})

    assert [result["id"] for result in batch["results"]] == [
        "brick",
        "asphalt",
        "glass",
    ]
    assert batch["errors"] == []
    assert len(batch["results"]) == 3
