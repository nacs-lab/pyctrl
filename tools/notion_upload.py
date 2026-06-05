"""Upload local images to a Notion page via the File Upload REST API + attach them as image blocks.

General-purpose lab-notebook image uploader. The claude.ai Notion *connector* can't upload local
files (URL embeds only), so this uses Notion's REST File Upload API: per image, create a file_upload
-> POST the bytes -> append an `image` block (optionally `after` a given block, so figures land under
the right note instead of at the page bottom). Multiple images attach in ONE ordered PATCH.

Auth (token is NEVER written to the repo; priority order):
  --token-file <path>  |  env NOTION_TOKEN_FILE=<path>  |  env NOTION_TOKEN=<secret>
Token = a Notion *internal integration* secret; the page (or an ANCESTOR — e.g. the year page) must
be shared with that integration (page -> ... -> Connections -> add it).

Page: --page accepts a Notion URL or a 32-hex id (default: env NOTION_PAGE).
Images: positional specs `PATH[::CAPTION]` (use `::` so Windows `C:\\` paths survive; caption
defaults to the filename), and/or a --manifest file with one `PATH[::CAPTION]` per line.

Examples:
  python notion_upload.py --page <url|id> --token-file T --list                  # find block ids
  python notion_upload.py --page <url|id> --token-file T fig1.png::"Caption" fig2.png
  python notion_upload.py --page <url|id> --token-file T --after <block-id> overlay.png::"..."
  python notion_upload.py --page <url|id> --token-file T --manifest figs.txt
"""
import argparse
import mimetypes
import os
import re
import sys

try:                                   # PowerShell console is cp1252; avoid UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NOTION_VERSION = os.environ.get("NOTION_VERSION", "2022-06-28")
API = "https://api.notion.com/v1"


def normalize_page_id(s):
    """Accept a Notion URL or a bare id (with/without dashes) -> dashed 8-4-4-4-12 uuid."""
    s = (s or "").strip().split("?")[0]
    hexes = re.findall(r"[0-9a-fA-F]{32}", s.replace("-", ""))
    raw = hexes[-1] if hexes else s.replace("-", "")
    if len(raw) == 32:
        return "-".join([raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]])
    return s


def parse_spec(spec):
    if "::" in spec:
        p, c = spec.split("::", 1)
        return p.strip(), c.strip()
    return spec.strip(), os.path.basename(spec.strip())


def load_token(args):
    path = args.token_file or os.environ.get("NOTION_TOKEN_FILE")
    if path:
        return open(path).read().strip()
    tok = os.environ.get("NOTION_TOKEN")
    if not tok:
        sys.exit("No token: pass --token-file PATH, or set NOTION_TOKEN_FILE / NOTION_TOKEN.")
    return tok.strip()


def main():
    ap = argparse.ArgumentParser(description="Upload local images to a Notion page (REST File Upload API).")
    ap.add_argument("--page", default=os.environ.get("NOTION_PAGE"),
                    help="Notion page URL or id (default: env NOTION_PAGE)")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--after", default=None, help="insert image block(s) after this block id")
    ap.add_argument("--manifest", default=None, help="file with one 'PATH[::CAPTION]' per line")
    ap.add_argument("--list", action="store_true", help="list the page's top-level blocks + exit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("images", nargs="*", help="PATH[::CAPTION] specs")
    args = ap.parse_args()

    if not args.page:
        sys.exit("No page: pass --page <url|id> or set env NOTION_PAGE.")
    try:
        import requests
    except ImportError:
        sys.exit("needs `requests` (present in the anaconda base / yb_analysis envs).")

    page = normalize_page_id(args.page)
    token = load_token(args)
    H = {"Authorization": "Bearer " + token, "Notion-Version": NOTION_VERSION}

    if args.list:
        r = requests.get("%s/blocks/%s/children?page_size=100" % (API, page), headers=H, timeout=30)
        if r.status_code >= 400:
            sys.exit("LIST FAILED %s: %s" % (r.status_code, r.text[:400]))
        for b in r.json().get("results", []):
            t = b.get("type")
            rich = b.get(t, {}).get("rich_text", []) if isinstance(b.get(t), dict) else []
            snip = "".join(x.get("plain_text", "") for x in rich)[:70].replace("\n", " ")
            print("%s  %-18s %s" % (b["id"], t, snip))
        return

    # gather image specs (positional + manifest)
    specs = list(args.images)
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    specs.append(ln)
    images = [parse_spec(s) for s in specs]
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit("No images given (positional PATH[::CAPTION] specs or --manifest).")

    print("page=%s version=%s after=%s -> %d image(s)"
          % (page, NOTION_VERSION, args.after, len(images)))
    blocks = []
    for path, caption in images:
        name = os.path.basename(path)
        if not os.path.isfile(path):
            print("  SKIP (missing): %s" % path); continue
        ctype = mimetypes.guess_type(path)[0] or "image/png"
        r = requests.post("%s/file_uploads" % API, headers=H,
                          json={"filename": name, "content_type": ctype}, timeout=30)
        if r.status_code >= 400:
            print("  CREATE FAILED %s: %s -> %s" % (name, r.status_code, r.text[:300])); continue
        up = r.json(); fid = up["id"]; send_url = up["upload_url"]
        with open(path, "rb") as fh:
            r = requests.post(send_url, headers=H, files={"file": (name, fh, ctype)}, timeout=180)
        if r.status_code >= 400:
            print("  SEND FAILED %s: %s -> %s" % (name, r.status_code, r.text[:300])); continue
        blocks.append({"object": "block", "type": "image",
                       "image": {"type": "file_upload", "file_upload": {"id": fid},
                                 "caption": [{"type": "text", "text": {"content": caption}}]}})
        print("  uploaded: %s" % name)
    if not blocks:
        print("nothing uploaded"); return
    body = {"children": blocks}              # one ordered PATCH (preserves order; `after` = mid-page)
    if args.after:
        body["after"] = args.after
    r = requests.patch("%s/blocks/%s/children" % (API, page),
                       headers={**H, "Content-Type": "application/json"}, json=body, timeout=60)
    if r.status_code >= 400:
        print("ATTACH FAILED: %s -> %s" % (r.status_code, r.text[:400])); return
    where = ("after " + args.after) if args.after else "at end of page"
    print("done: attached %d image block(s) %s" % (len(blocks), where))


if __name__ == "__main__":
    main()
