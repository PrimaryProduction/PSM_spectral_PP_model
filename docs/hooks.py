"""MkDocs hook that renders numpydoc/reST markup found in the docstrings.

The docstrings follow the numpydoc convention, which uses reStructuredText
roles. mkdocstrings renders docstrings as Markdown, so after rendering
(outside ``<pre>`` code blocks):

* ``:math:`x``` (rendered as ``:math:<code>x</code>``) becomes inline MathJax,
* ``:func:``, ``:class:``, ``:mod:``, ``:data:``, ``:attr:`` roles become
  plain inline code,
* ``.. [1]`` citation labels and ``[1]_`` references become ``[1]``,
* ``.. warning::`` becomes a bold "Warning:" label.
"""

import re

_MATH = re.compile(r":math:<code>(.*?)</code>", re.S)
_ROLE = re.compile(r":(?:func|class|mod|data|attr|meth|obj):(<code>)")
_CITE_FIRST = re.compile(r"<p>\.\. \[(\d+)\]")
_CITE = re.compile(r"\s*\.\. \[(\d+)\]")
_CITE_REF = re.compile(r"\[(\d+)\]_")
_WARNING = re.compile(r"\.\. warning::")


_PRE = re.compile(r"(<pre\b.*?</pre>)", re.S)


def _convert(html):
    html = _MATH.sub(r'<span class="arithmatex">\\(\1\\)</span>', html)
    html = _ROLE.sub(r"\1", html)
    html = _CITE_FIRST.sub(r"<p>[\1]", html)
    html = _CITE.sub(r"<br>[\1]", html)
    html = _CITE_REF.sub(r"[\1]", html)
    html = _WARNING.sub("<strong>Warning:</strong>", html)
    return html


def on_page_content(html, page, config, files):
    # Leave code blocks (including the "Source code" panels) untouched.
    parts = _PRE.split(html)
    return "".join(p if i % 2 else _convert(p) for i, p in enumerate(parts))
