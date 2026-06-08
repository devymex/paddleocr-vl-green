from __future__ import annotations

import html as html_lib
from typing import List

from .config import DROP_LABELS

TITLE_LABELS = {"doc_title", "title"}
HEADING_LABELS = {"paragraph_title", "section_title", "heading"}
TABLE_LABELS = {"table"}
FORMULA_LABELS = {"formula"}
PIC_LABELS = {"image", "figure"}
CAPTION_LABELS = {"caption", "figure_caption", "table_caption", "vision_footnote"}
CODE_LABELS = {"code"}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['\\\\(','\\\\)'], ['$','$']], displayMath: [['\\\\[','\\\\]'], ['$$','$$']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
          max-width: 960px; margin: 40px auto; padding: 0 20px; line-height: 1.8; color: #222; }}
  h1.doc-title {{ text-align: center; font-size: 1.6em; margin-bottom: 1em; }}
  h2.paragraph-title {{ font-size: 1.2em; margin-top: 1.5em; border-left: 4px solid #4a90d9; padding-left: 8px; }}
  .text-block {{ margin: 0.6em 0; }}
  .table-block {{ overflow-x: auto; margin: 1em 0; }}
  .table-block table {{ border-collapse: collapse; width: 100%; word-wrap: break-word; }}
  .table-block td, .table-block th {{ border: 1px solid #999; padding: 6px 10px; }}
  .formula-block {{ margin: 1em 0; text-align: center; font-size: 1.05em; }}
  .caption {{ color: #666; font-size: 0.9em; text-align: center; }}
  .image-block {{ text-align: center; color: #888; }}
  .code-block {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
  hr.page-sep {{ border: none; border-top: 2px dashed #ccc; margin: 2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def block_to_html(block: dict) -> str:
    label = block["label"].lower()
    content = block["content"]
    image_base64 = block.get("image_base64")

    if label in TABLE_LABELS:
        return f'<div class="block table-block">\n{content}\n</div>'
    if label in FORMULA_LABELS:
        return f'<div class="block formula-block">\\[{html_lib.escape(content)}\\]</div>'
    if label in PIC_LABELS:
        if image_base64:
            return f'<div class="block image-block"><img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto;" /></div>'
        else:
            return f'<div class="block image-block"><p>[图片: {html_lib.escape(content)}]</p></div>'
    if label in TITLE_LABELS:
        return f'<h1 class="block doc-title">{html_lib.escape(content)}</h1>'
    if label in HEADING_LABELS:
        return f'<h2 class="block paragraph-title">{html_lib.escape(content)}</h2>'
    if label in CAPTION_LABELS:
        return f'<p class="block caption">{html_lib.escape(content)}</p>'
    if label in CODE_LABELS:
        return f'<pre class="block code-block"><code>{html_lib.escape(content)}</code></pre>'
    return f'<p class="block text-block">{html_lib.escape(content)}</p>'


def render_html(blocks: List[dict], title: str = "document") -> str:
    """Render a list of blocks to a complete HTML page."""
    parts = []
    for b in blocks:
        label = b["label"]
        content = b["content"]
        if not content or label in DROP_LABELS:
            continue
        parts.append(block_to_html(b))
    body = '<div class="page" id="page-1">\n' + "\n".join(parts) + "\n</div>"
    return HTML_TEMPLATE.format(title=html_lib.escape(title), body=body)
