"""
DFS-based STEP reserialization — core innovation from STEP-LLM paper Section 3.1.

Raw STEP files are graph-structured: #1 might reference #453 which references #789.
LLMs must track these long-range dependencies.  DFS reserialization converts the
entity DAG into a locality-preserving linear sequence where related entities are
always adjacent, then renumbers IDs sequentially so there are no gaps or jumps.

Three transformations applied:
  1. Sequential renumbering (no long-range #ID references)
  2. Normalized float precision (6 decimal places, reduces token count)
  3. CoT-style branch annotations before each root entity
     (/* [BRANCH] depth=D children=C */)
"""

import re

from data.step_parser import _mask_string_literals  # C9: shared string-literal masking


def reserialize(header: str, entities: dict, referenced_by: dict) -> str:
    """
    Full DFS reserialization as described in STEP-LLM paper Section 3.1.

    Args:
        header: STEP header block (everything before DATA;)
        entities: {id: {"type": str, "args": str, "refs": List[int]}}
        referenced_by: {id: Set[int]} — reverse reference index

    Returns:
        Complete, valid STEP file string with DFS-ordered entities.
    """
    # Step 1: Find roots = entities not referenced by anything
    all_ids = set(entities.keys())
    roots = sorted(all_ids - set(referenced_by.keys()))  # sorted for determinism

    # Step 2: DFS traversal — builds output_order (list of old IDs in DFS order)
    visited = set()
    output_order = []

    # W5: compute branch stats DURING the main traversal so the count matches
    # what's actually serialized under each root. The old branch_stats() used a
    # fresh visited set, double-counting entities shared between roots.
    root_stats: dict[int, tuple[int, int]] = {}
    for root in roots:
        start_len = len(output_order)
        # depth-tracking DFS for this root
        stack = [(root, 0)]
        max_depth = 0
        while stack:
            node_id, depth = stack.pop()
            if node_id in visited or node_id not in entities:
                continue
            visited.add(node_id)
            output_order.append(node_id)
            max_depth = max(max_depth, depth)
            for child_id in reversed(entities[node_id]["refs"]):
                if child_id not in visited:
                    stack.append((child_id, depth + 1))
        count = len(output_order) - start_len
        if count > 0:
            root_stats[root] = (max_depth, count)

    # Catch unreachable entities (rare isolated nodes). DFS from each one
    # so a stranded subgraph (e.g. cycle A↔B with B→C) keeps locality
    # instead of being flattened in arbitrary old-ID order.
    for orphan_root in sorted(entities.keys()):
        if orphan_root in visited:
            continue
        stack = [(orphan_root, 0)]
        while stack:
            node_id, depth = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            output_order.append(node_id)
            for child_id in reversed(entities[node_id]["refs"]):
                if child_id not in visited:
                    stack.append((child_id, depth + 1))

    # Step 3: Sequential renumbering map: old_id → new_id (1-indexed)
    id_map = {old: (new_idx + 1) for new_idx, old in enumerate(output_order)}

    # C5/W4/W9/S6: float normalization that preserves scientific notation.
    # The old f"{x:.6f}" turns 1.1234567E-12 into 0.000000 (precision destroyed)
    # and 1.1234567E+10 into 11234567000.000000 (token bloat). Round the
    # mantissa, keep the exponent — same semantics as round_step_numbers.py.
    # ISO 10303-21 §6.4.3 REAL: 1.5, 1., .5, 1.5E-3, 1.5e-3.
    _float_re = re.compile(r"-?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?")

    def _normalize_float(m: re.Match) -> str:
        s = m.group(0)
        if "E" in s or "e" in s:
            mant, _, exp = s.replace("e", "E").partition("E")
            return f"{round(float(mant), 6):.6f}E{exp}"
        return f"{round(float(s), 6):.6f}"

    # C9: only renumber #N tokens that are OUTSIDE string literals.
    _ref_re = re.compile(r"#(\d+)")

    def _rewrite_outside_strings(args: str) -> str:
        out, last = [], 0
        for sm in re.finditer(r"'(?:[^']|'')*'", args):
            seg = args[last:sm.start()]
            # C7: hard KeyError on unmapped ref. The old .get(..., old_id)
            # fallback emitted dangling references for entities that
            # step_parser silently dropped — invalid STEP in the training label.
            seg = _ref_re.sub(lambda m: f"#{id_map[int(m.group(1))]}", seg)
            seg = _float_re.sub(_normalize_float, seg)
            out.append(seg)
            out.append(sm.group(0))  # string literal: pass through verbatim
            last = sm.end()
        seg = args[last:]
        seg = _ref_re.sub(lambda m: f"#{id_map[int(m.group(1))]}", seg)
        seg = _float_re.sub(_normalize_float, seg)
        out.append(seg)
        return "".join(out)

    # Step 5: Rewrite each entity with new IDs + normalized floats
    def rewrite(eid):
        e = entities[eid]
        new_id = id_map[eid]
        new_args = _rewrite_outside_strings(e["args"])
        # C6: complex entities have type="" and use #N = ( ... ); syntax.
        if e["type"]:
            return f"#{new_id} = {e['type']}({new_args});"
        return f"#{new_id} = ({new_args});"

    # Step 6: Assemble output with CoT branch annotations.
    # C8: ENDSEC; before DATA; is now preserved by step_parser.py in the
    # header string. If an old/external caller passes a header without it,
    # inject it here so the output is always ISO-10303-21 valid.
    if header and "ENDSEC;" not in header:
        lines = [header, "ENDSEC;", "DATA;"]
    else:
        lines = [header, "DATA;"]
    for eid in output_order:
        if eid in root_stats:
            depth, children = root_stats[eid]
            lines.append(f"/* [BRANCH] depth={depth} children={children} */")
        lines.append(rewrite(eid))
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")

    return "\n".join(lines)
