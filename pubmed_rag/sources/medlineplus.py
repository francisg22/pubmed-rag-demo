"""Parse a MedlinePlus Health Topics bulk XML file into per-topic Markdown.

MedlinePlus health-topic summaries are produced by the U.S. National Library of
Medicine and are in the PUBLIC DOMAIN, which makes them a clean grounding source
for the general-knowledge corpus. (Do NOT confuse them with the linked A.D.A.M.
Medical Encyclopedia, which is copyrighted and is not in this XML.)

Get the XML from https://medlineplus.gov/xml.html (file named like
`mplus_topics_YYYY-MM-DD.xml`). This module turns each English health topic into a
clean `.md` file under the general-knowledge corpus dir, ready for
`python -m pubmed_rag ingest --corpus general`.

    python -m pubmed_rag.sources.medlineplus \
        --xml data/general_docs/mplus_topics_2026-07-07.xml \
        --out data/general_docs/medlineplus            # English only by default

Optional: --groups "Bones, Joints and Muscles" "Cancers"   (scope to specialties)
"""
import argparse
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")
_WS_LINE = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")
_SLUG = re.compile(r"[^a-z0-9]+")


def _clean_summary(raw: str) -> str:
    """MedlinePlus full-summary is a fragment of HTML. Convert the block structure
    to readable Markdown-ish text: paragraphs -> blank lines, <li> -> bullets."""
    if not raw:
        return ""
    t = html.unescape(raw)
    t = re.sub(r"<li[^>]*>", "\n- ", t, flags=re.I)
    t = re.sub(r"</p\s*>", "\n\n", t, flags=re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = _TAG.sub("", t)                      # drop remaining tags (links, spans, ...)
    t = html.unescape(t)                     # unescape anything the tags hid
    t = _WS_LINE.sub(" ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = _MULTINL.sub("\n\n", t)
    return t.strip()


def _slug(title: str) -> str:
    return _SLUG.sub("-", (title or "topic").lower()).strip("-")[:80] or "topic"


def parse(xml_path: str, out_dir: str, lang: str = "English",
          groups: list[str] | None = None, min_tokens: int = 40) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    group_filter = {g.lower() for g in groups} if groups else None

    written = skipped_lang = skipped_group = skipped_short = 0
    seen: set[str] = set()

    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != "health-topic":
            continue
        try:
            if (el.get("language") or "").strip().lower() != lang.lower():
                skipped_lang += 1
                continue
            topic_groups = [(g.text or "").strip() for g in el.findall("group") if (g.text or "").strip()]
            if group_filter and not any(g.lower() in group_filter for g in topic_groups):
                skipped_group += 1
                continue
            title = (el.get("title") or "").strip()
            body = _clean_summary(el.findtext("full-summary") or "")
            if not body or len(body) < min_tokens * 3:   # ~3 chars/token floor
                skipped_short += 1
                continue
            also = [
                (a.text or "").strip()
                for a in el.findall("also-called")
                if (a.text or "").strip()
            ]

            parts = [f"# {title}", ""]
            if also:
                parts += [f"*Also called: {', '.join(also)}*", ""]
            parts += [body, ""]
            if topic_groups:
                parts += [f"_Category: {', '.join(topic_groups)}_"]
            parts += [
                "_Source: MedlinePlus, U.S. National Library of Medicine (public domain). "
                f"Topic: {title}._"
            ]

            name = _slug(title)
            fname = name + ".md"
            i = 2
            while fname in seen:               # avoid collisions on duplicate slugs
                fname = f"{name}-{i}.md"
                i += 1
            seen.add(fname)
            (out / fname).write_text("\n".join(parts), encoding="utf-8")
            written += 1
        finally:
            el.clear()

    return {
        "written": written,
        "skipped_other_language": skipped_lang,
        "skipped_group_filtered": skipped_group,
        "skipped_too_short": skipped_short,
        "out_dir": str(out),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        prog="pubmed_rag.sources.medlineplus",
        description="Parse MedlinePlus health-topic XML into per-topic Markdown for the general corpus.",
    )
    p.add_argument("--xml", required=True, help="path to mplus_topics_YYYY-MM-DD.xml")
    p.add_argument("--out", required=True, help="output dir (e.g. data/general_docs/medlineplus)")
    p.add_argument("--lang", default="English", help="topic language to keep (default English)")
    p.add_argument("--groups", nargs="*", help="only keep topics in these groups (exact names)")
    p.add_argument("--min-tokens", type=int, default=40, help="skip summaries shorter than this")
    args = p.parse_args()

    stats = parse(args.xml, args.out, lang=args.lang, groups=args.groups, min_tokens=args.min_tokens)
    print(f"Wrote {stats['written']} topics -> {stats['out_dir']}")
    print(f"  skipped: {stats['skipped_other_language']} other-language, "
          f"{stats['skipped_group_filtered']} group-filtered, "
          f"{stats['skipped_too_short']} too short")
    print("\nNext: dry-run, then embed:")
    print("  python -m pubmed_rag ingest --corpus general --dry-run")
    print("  ALLOW_EMBEDDING=1 python -m pubmed_rag ingest --corpus general")


if __name__ == "__main__":
    main()
