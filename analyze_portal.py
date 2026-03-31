from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"


def find_all(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def main() -> None:
    html = INDEX_FILE.read_text(encoding="utf-8")
    lines = html.splitlines()

    section_ids = find_all(r'<section class="view(?: active)?" id="([^"]+)"', html)
    all_hrefs = find_all(r'href="([^"]+)"', html)
    external_links = [href for href in all_hrefs if href.startswith("http")]
    mailto_links = [href for href in all_hrefs if href.startswith("mailto:")]
    github_links = [href for href in external_links if "github.com" in href.lower()]

    repo_pairs = [
        f"{handle}/{repo}"
        for handle, repo in re.findall(
            r'handle:\s*"([^"]+)"\s*,\s*repo:\s*"([^"]+)"',
            html,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    ]
    contact_labels = find_all(r'contactLabel:\s*"([^"]+)"', html)

    domain_counts = Counter(
        re.sub(r"^https?://", "", href).split("/")[0].lower()
        for href in external_links
    )

    print("Portal Analysis")
    print("=" * 15)
    print(f"File: {INDEX_FILE.name}")
    print(f"Lines: {len(lines)}")
    print(f"View sections: {len(section_ids)} -> {', '.join(section_ids)}")
    print(f"External links: {len(external_links)}")
    print(f"Mailto links: {len(mailto_links)}")
    print(f"GitHub links: {len(github_links)}")
    print()

    print("Project Mentions")
    print("-" * 16)
    for repo in unique_preserve_order(repo_pairs):
        print(f"- {repo}")
    if not repo_pairs:
        print("- none detected")
    print()

    print("Contact Labels")
    print("-" * 14)
    for label in unique_preserve_order(contact_labels):
        print(f"- {label}")
    if not contact_labels:
        print("- none detected")
    print()

    print("Link Domains")
    print("-" * 12)
    for domain, count in sorted(domain_counts.items()):
        print(f"- {domain}: {count}")
    if not domain_counts:
        print("- none detected")


if __name__ == "__main__":
    main()
