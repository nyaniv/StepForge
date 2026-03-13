"""
Parse a STEP file into an entity DAG for DFS reserialization.

A STEP DATA SECTION is a flat list of numbered entities that cross-reference
each other by #ID.  This module extracts the header, entity dict, and reverse
reference index required by dfs_reserializer.py.

Handles OCC-generated STEP files which have:
  - A HEADER section ending with ENDSEC; (before DATA;)
  - Multi-line entity definitions (continuation lines until semicolon)
"""

import re


def _parse_lines(lines):
    """
    Core parser shared by parse_step() and parse_step_from_string().

    Returns (header_str, entities, referenced_by).

    Handles:
      1. HEADER...ENDSEC; section before DATA; — skip ENDSEC; until in DATA
      2. Multi-line entity definitions — accumulate until line ends with ';'
    """
    header_lines = []
    entities = {}
    referenced_by = {}
    in_data = False
    pending = ""   # accumulates multi-line entity text

    for raw_line in lines:
        line = raw_line.strip()

        if not in_data:
            if line == "DATA;":
                in_data = True
                continue
            # ENDSEC; ends the HEADER section — skip it, keep collecting header
            if line in ("ENDSEC;", "END-ISO-10303-21;"):
                continue
            header_lines.append(line)
            continue

        # Inside DATA section
        if line in ("ENDSEC;", "END-ISO-10303-21;"):
            break   # end of DATA section

        # Accumulate multi-line entities
        if pending:
            # Join with a space; strip leading whitespace from continuation lines
            pending = pending + " " + line
        else:
            pending = line

        # A complete entity ends with ';'
        if not pending.endswith(";"):
            continue

        entity_line = pending
        pending = ""

        m = re.match(r"#(\d+)\s*=\s*(\w+)\s*\((.+)\)\s*;$", entity_line, re.DOTALL)
        if not m:
            continue

        eid   = int(m.group(1))
        etype = m.group(2)
        eargs = m.group(3)

        refs = [int(r) for r in re.findall(r"#(\d+)", eargs)]
        entities[eid] = {"type": etype, "args": eargs, "refs": refs}
        for r in refs:
            referenced_by.setdefault(r, set()).add(eid)

    return "\n".join(header_lines), entities, referenced_by


def parse_step(filepath: str) -> tuple:
    """
    Parse a STEP file into its constituent parts.

    Returns:
        header (str): everything before 'DATA;' — preserved unchanged in output
        entities (dict): {entity_id (int): {"type": str, "args": str, "refs": List[int]}}
        referenced_by (dict): {entity_id (int): Set[int]} — reverse reference index
    """
    with open(filepath, errors="replace") as f:
        return _parse_lines(f)


def parse_step_from_string(content: str) -> tuple:
    """
    Same as parse_step but operates on an in-memory STEP string.
    Used by filter_dataset.py after pairing captions.
    """
    return _parse_lines(content.splitlines())
