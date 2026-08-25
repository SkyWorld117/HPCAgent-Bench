# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which numpy calls the Fortran backend renders as an intrinsic instead of a loop nest.

Lowering expands a numpy call to an explicit loop for EVERY target, which is right for C -- it has
no array intrinsics -- and throws away the compiler's own implementation in Fortran. ``SUM``,
``MAXVAL``, ``PRODUCT``, ``NORM2`` and the rest are vectorized by the compiler and carry their
meaning in their name. So the Fortran driver hands :func:`numpyto_common.lowering.lower` the
predicate below, the calls named here survive lowering untouched, and ``_emit_call`` renders them --
that code already exists, it was simply unreachable because nothing ever reached emit unexpanded.

WHOLE-ARRAY reductions only. They produce a SCALAR, so no downstream pass has to shape or allocate
a result, which is what keeps this a local change. A call carrying an ``axis`` (or any other
argument) still lowers to loops: the intrinsic's ``dim=`` counts in Fortran's axis order, not
numpy's, and a wrong ``dim`` is a silently wrong answer rather than a refusal. ``argmax``/``argmin``
are held back for the neighbouring reason -- ``MAXLOC`` is 1-based where numpy is 0-based.
"""
import ast
from typing import Dict, Optional, Tuple

#: ``(module, attr)`` keys whose Fortran intrinsic reduces the whole array to a scalar AND agrees
#: with numpy on a FLOATING operand. The set is short on purpose -- each name kept out is a
#: measured disagreement, not an unwritten rendering:
#:
#: * ``max`` / ``min``: numpy PROPAGATES NaN, ``MAXVAL`` / ``MINVAL`` do not. The loop lowering
#:   emits the NaN-faithful form; the intrinsic would silently answer with a non-NaN element.
#: * ``all`` / ``any`` / ``count_nonzero``: the result is ``LOGICAL`` (or an integer count) while
#:   the hoisted temp is declared ``real``, and ``COUNT(m /= 0)`` on a ``LOGICAL`` operand is not
#:   even a legal comparison. Both need result-dtype plumbing that does not exist yet.
WHOLE_ARRAY_REDUCTIONS = frozenset({
    ("np", "sum"),
    ("np", "prod"),
    ("np", "mean"),
    ("np", "linalg.norm"),
})

#: Elementwise ops with a Fortran intrinsic that takes whole arrays. ``MERGE`` evaluates BOTH of its
#: branches, which is what ``np.where`` does too -- so unlike a guarded division, the eager form is
#: the faithful one here.
ELEMENTWISE_INTRINSICS = frozenset({("np", "where")})

#: ``(module, attr)`` keys whose Fortran intrinsic also takes a ``dim=``, reducing ONE axis.
#: ``norm`` is absent: Fortran has no per-axis 2-norm, and ``median`` has no intrinsic at all.
AXIS_REDUCTIONS = frozenset({
    ("np", "sum"),
    ("np", "prod"),
    ("np", "mean"),
})


def literal_axis(call: ast.Call) -> Optional[int]:
    """``call``'s axis as a single literal int, or ``None`` when it is anything else.

    ``keepdims`` decides the result RANK and no ``dim=`` reduction preserves it, so its presence
    declines outright. A TUPLE axis is declined too -- Fortran's ``dim=`` takes one axis, and the
    nested form that would express a tuple is not what this maps. numpy accepts the axis
    positionally as well, which is why the positional slot is read and not only the keyword.
    """
    if any(k.arg in ("keepdims", "dtype", "out", "where", "initial") for k in call.keywords):
        return None
    axis = next((k.value for k in call.keywords if k.arg == "axis"), None)
    if axis is None and len(call.args) > 1:
        axis = call.args[1]
    if isinstance(axis, ast.Constant) and isinstance(axis.value, int) and not isinstance(axis.value, bool):
        return axis.value
    return None


#: Element types on which the kept intrinsics agree with numpy exactly. INTEGER is excluded because
#: the two disagree on overflow -- numpy's int32 sum wraps, Fortran's does not -- and because an
#: integer ``mean`` would become integer division. BOOLEAN is excluded because ``SUM`` of a
#: ``LOGICAL`` is not valid Fortran at all.
_FLOAT_KINDS = ("float", "double", "real", "complex")

#: numpy op -> which argument slots must be a floating array for the claim to be safe. Default is
#: the single operand at 0; ``where`` selects BETWEEN its second and third, and its first is a mask.
_FLOAT_SLOTS: Dict[str, Tuple[int, ...]] = {"where": (1, 2)}


def _slot_is_float(node: ast.expr, dtypes: Dict[str, str]) -> bool:
    """A float array Name, or a float literal -- the two things ``MERGE`` can hold side by side."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if not isinstance(node, ast.Name):
        return False
    dtype = dtypes.get(node.id)
    return bool(dtype) and any(dtype.startswith(k) for k in _FLOAT_KINDS)


def operand_is_float(call: ast.Call, dtypes: Dict[str, str]) -> bool:
    """True only on POSITIVE evidence that the operand is a floating array.

    Absence of a dtype is not evidence of a float: the lowering context tags integers and leaves
    everything else untagged, so an unknown name could be a boolean mask. Declining an unknown costs
    a loop nest; claiming one wrongly costs a wrong answer or a Fortran compile error.
    """
    attr = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    slots = _FLOAT_SLOTS.get(attr, (0, ))
    if len(call.args) <= max(slots):
        return False
    return all(_slot_is_float(call.args[i], dtypes) for i in slots)


def renders_natively(key: Tuple[str, str], call: ast.Call, shapes: Dict[str, Tuple[str, ...]],
                     dtypes: Dict[str, str]) -> bool:
    """True when Fortran has an intrinsic for ``call`` and lowering should leave it alone.

    Decided HERE, with the shape table in hand, rather than at emit: once lowering hands the call
    on there is no loop nest left to fall back to, so a claim the emitter cannot honour turns a
    slower kernel into a refused one. Hence the operand must be a bare Name of known rank and the
    axis must be in range for it.

    Whole-array form: one operand, nothing else. ``axis``, ``keepdims``, ``dtype``, ``out`` and
    ``norm``'s ``ord`` each ask for something the whole-array intrinsic does not answer, and numpy
    takes the axis positionally too -- so for those keys a second argument of any kind declines.
    """
    if not operand_is_float(call, dtypes):
        return False
    if key in ELEMENTWISE_INTRINSICS and len(call.args) == 3 and not call.keywords:
        return True
    if key in WHOLE_ARRAY_REDUCTIONS and len(call.args) == 1 and not call.keywords:
        return True
    if key not in AXIS_REDUCTIONS:
        return False
    axis = literal_axis(call)
    if axis is None:
        return False
    operand = call.args[0]
    if not isinstance(operand, ast.Name):
        return False
    shape = shapes.get(operand.id)
    if not shape:
        return False
    rank = len(shape)
    return -rank <= axis < rank
