"""Inject the smash.glb model into landing_template.html as a base64 data URI.

Lets the landing page work via plain file:// (no CORS, no server needed) at the
cost of a ~10MB HTML file.
"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GLB = os.path.join(ROOT, "models", "smash.glb")
TEMPLATE = os.path.join(ROOT, "web", "landing_template.html")
OUT = os.path.join(ROOT, "web", "landing.html")
PLACEHOLDER = "__SMASH_GLB__"


def main():
    if not os.path.exists(GLB):
        raise SystemExit(f"missing {GLB}")
    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"missing {TEMPLATE}")

    print(f"reading {GLB} ({os.path.getsize(GLB)//1024} KB)")
    with open(GLB, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    data_uri = f"data:model/gltf-binary;base64,{b64}"

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if PLACEHOLDER not in html:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in template")
    out = html.replace(PLACEHOLDER, data_uri)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
