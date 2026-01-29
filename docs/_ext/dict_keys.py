"""
Credits to ChatGPT 5.2
"""
from __future__ import annotations

import importlib
from typing import Any, Iterable, List, Optional, Tuple

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import ViewList

from sphinx.pycode import ModuleAnalyzer

def _safe_sorted_items(d: Any) -> List[Tuple[Any, Any]]:
    """Try to sort dict items by key (as str). Fall back to insertion order."""
    items = list(d.items())
    try:
        items.sort(key=lambda kv: str(kv[0]))
    except Exception:
        pass
    return items

def _safe_sorted_keys(d: Any) -> List[Any]:
    keys = list(d.keys())
    try:
        keys.sort(key=lambda k: str(k))
    except Exception:
        pass
    return keys

def _make_table(
    rows: Iterable[Tuple[str, str]],
    headers: Tuple[str, ...],
    colwidths: Optional[Tuple[int, ...]] = None,
) -> nodes.table:
    """
    Build a docutils table node.

    headers: tuple of column header strings
    rows: iterable of tuples of cell strings, must match len(headers)
    colwidths: optional relative widths (same length as headers)
    """
    ncols = len(headers)
    table = nodes.table()
    tgroup = nodes.tgroup(cols=ncols)
    table += tgroup

    if colwidths is None:
        # simple default: equal widths
        colwidths = tuple([1] * ncols)
    if len(colwidths) != ncols:
        raise ValueError("colwidths must match number of headers")

    for w in colwidths:
        tgroup += nodes.colspec(colwidth=w)

    thead = nodes.thead()
    tgroup += thead
    hrow = nodes.row()
    for h in headers:
        entry = nodes.entry()
        entry += nodes.paragraph(text=h)
        hrow += entry
    thead += hrow

    tbody = nodes.tbody()
    tgroup += tbody

    for row in rows:
        if len(row) != ncols:
            raise ValueError("row length must match number of headers")
        rnode = nodes.row()
        for cell in row:
            entry = nodes.entry()
            entry += nodes.paragraph(text=cell)
            rnode += entry
        tbody += rnode

    return table


class DictKeys(Directive):
    required_arguments = 1

    option_spec = {
        "key-header": directives.unchanged,     # default "Key"
        "value-header": directives.unchanged,   # default "Value"
    }

    def _import_target(self, target: str) -> Any:
        module_path, obj_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, obj_name)

    def _parse_rst(self, rst_lines):
        vl = ViewList()
        for line in rst_lines:
            vl.append(line, source="dict-keys")
        section = nodes.section()
        self.state.nested_parse(vl, self.content_offset, section)
        return list(section.children)

    def _get_attr_doc(self, module_name: str, attr_name: str) -> str | None:
        """
        Returns doc-comment text (from '#:' comments) for a module attribute,
        or None if not found.
        """
        try:
            analyzer = ModuleAnalyzer.for_module(module_name)
            analyzer.analyze()
            # keys are typically tuples like (classname, attrname); module vars use classname=""
            docs = analyzer.attr_docs  # populated by analyze()
        except Exception:
            return None

        # Try common keys Sphinx uses for module-level attrs
        for key in (("", attr_name), (None, attr_name), ("module", attr_name)):
            if key in docs:
                # docs[key] is usually a list of lines
                text = "\n".join(docs[key]).strip()
                return text or None
        return None

    def run(self) -> List[nodes.Node]:
        target = self.arguments[0].strip()  # e.g. my_package.MAP
        module_path, obj_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = self._import_target(target)

        if not hasattr(obj, "keys"):
            error = self.state_machine.reporter.error(
                f"dict-keys target {target!r} is not dict-like (missing .keys())",
                line=self.lineno,
            )
            return [error]

        container = nodes.container()
        container["classes"].append("dict-keys")

        # Clickable header: :attr:`~dymad.blah.MAP`
        container += self._parse_rst([f":attr:`~{target}`", ""])

        # Optional doc text
        doc = self._get_attr_doc(module.__name__, obj_name)
        if doc:
            # allow RST markup inside the doc comment if you want
            container += self._parse_rst([doc, ""])
        else:
            # optional: omit silently; or show a small note if you prefer
            pass

        # Table: keys only or key/value
        key_header = self.options.get("key-header", "Key")
        value_header = self.options.get("value-header", "Value")

        items = _safe_sorted_items(obj)
        rows = []
        for k, v in items:
            _v = v
            if isinstance(_v, (tuple, list)):
                if callable(_v[0]):
                    _v = ", ".join(v.__name__ for v in _v)
            else:
                _v = _v.__name__
            rows.append((str(k), _v))

        # Relative widths: key narrower, value wider
        table = _make_table(rows, headers=(key_header, value_header), colwidths=(30, 70))

        container += table
        return [container]


def setup(app):
    app.add_directive("dict-keys", DictKeys)
    return {"version": "1.0.0", "parallel_read_safe": True, "parallel_write_safe": True}
