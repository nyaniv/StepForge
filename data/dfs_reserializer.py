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
import sys


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

    def dfs(node_id):
        if node_id in visited or node_id not in entities:
            return
        visited.add(node_id)
        output_order.append(node_id)
        for child_id in entities[node_id]["refs"]:
            dfs(child_id)  # recurse depth-first

    sys.setrecursionlimit(100000)
    for root in roots:
        dfs(root)

    # Catch unreachable entities (rare isolated nodes)
    for eid in sorted(entities.keys()):
        if eid not in visited:
            output_order.append(eid)

    # Step 3: Sequential renumbering map: old_id → new_id (1-indexed)
    id_map = {old: (new_idx + 1) for new_idx, old in enumerate(output_order)}

    # Step 4: Compute branch stats for CoT annotations
    def branch_stats(root_id):
        stack = [(root_id, 0)]
        b_visited = set()
        max_depth, count = 0, 0
        while stack:
            nid, depth = stack.pop()
            if nid in b_visited or nid not in entities:
                continue
            b_visited.add(nid)
            max_depth = max(max_depth, depth)
            count += 1
            for child in entities[nid]["refs"]:
                stack.append((child, depth + 1))
        return max_depth, count

    root_stats = {r: branch_stats(r) for r in roots}

    # Step 5: Rewrite each entity with new IDs + normalized floats
    def rewrite(eid):
        e = entities[eid]
        new_id = id_map[eid]
        # Replace all #OLD with #NEW in args
        new_args = re.sub(
            r"#(\d+)",
            lambda m: f"#{id_map.get(int(m.group(1)), int(m.group(1)))}",
            e["args"]
        )
        # Normalize floats to 6 decimal places (reduces token count)
        new_args = re.sub(
            r"(-?\d+\.\d{7,}E?[+-]?\d*)",
            lambda m: f"{float(m.group(1)):.6f}",
            new_args
        )
        return f"#{new_id} = {e['type']}({new_args});"

    # Step 6: Assemble output with CoT branch annotations
    lines = [header, "DATA;"]
    for eid in output_order:
        if eid in root_stats:
            depth, children = root_stats[eid]
            lines.append(f"/* [BRANCH] depth={depth} children={children} */")
        lines.append(rewrite(eid))
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")

    return "\n".join(lines)
