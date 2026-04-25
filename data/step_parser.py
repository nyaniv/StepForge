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

# C9: STEP string literals can contain '#42' (e.g. PRODUCT('Bracket #42 rev B',...)).
# Mask them before extracting #N references so part-number text isn't mistaken
# for a graph edge. STEP escapes single quotes by doubling ('').
_STR_LIT_RE = re.compile(r"'(?:[^']|'')*'")


def _mask_string_literals(text: str) -> str:
    """Replace string literal contents with underscores, preserving span lengths."""
    return _STR_LIT_RE.sub(lambda m: "_" * len(m.group()), text)


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
            # C8: keep ENDSEC; in the header so dfs_reserializer.py can emit
            # a structurally valid HEADER;...ENDSEC; DATA;...ENDSEC; file.
            # Only END-ISO-10303-21; is dropped (it goes at the very end).
            if line == "END-ISO-10303-21;":
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

        # A complete entity ends with ';'. ISO 10303-21 forbids newlines in
        # string literals so this is correct on conformant files; mask anyway
        # so a malformed multi-line literal can't cause a mid-entity false split.
        if not _mask_string_literals(pending).endswith(";"):
            continue

        entity_line = pending
        pending = ""

        m = re.match(r"#(\d+)\s*=\s*(\w+)\s*\((.*)\)\s*;$", entity_line, re.DOTALL)
        if m:
            eid   = int(m.group(1))
            etype = m.group(2)
            eargs = m.group(3)
        else:
            # C6: complex entities (#3 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) ... );)
            # have no identifier before '(' so the simple regex above can't match.
            # OCC always emits these for units/context — silently dropping them
            # produces dangling refs in the training label.
            mc = re.match(r"#(\d+)\s*=\s*\((.*)\)\s*;$", entity_line, re.DOTALL)
            if mc:
                eid   = int(mc.group(1))
                etype = ""  # complex entity: no single type name
                eargs = mc.group(2)
            elif entity_line.startswith("#"):
                # C10: hard-fail instead of silent continue. filter_dataset.py
                # wraps this in try/except, so a raise here correctly excludes
                # the file from training rather than passing through corrupt.
                raise ValueError(
                    f"step_parser: unparseable entity line: {entity_line[:120]!r}"
                )
            else:
                continue  # blank/comment lines inside DATA — harmless

        # C9: extract refs from a string-literal-masked copy.
        masked_args = _mask_string_literals(eargs)
        refs = [int(r) for r in re.findall(r"#(\d+)", masked_args)]
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
