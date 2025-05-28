#!/usr/bin/env python3
"""
fetch_from_repo_links.py

Download every file from a public GitHub repo, then print an ASCII tree
with each entry's raw.githubusercontent.com URL on the same line.

Usage:
    python fetch_from_repo_links.py <owner/repo> [branch] [dest_dir]
Example:
    python fetch_from_repo_links.py MagikEyeJp/SensorEvalGui main

GitHub API rate-limit: 60 req/h (unauthenticated). Script uses 1 API call
+ N raw downloads, so normally OK.
"""
import pathlib, requests, sys


def build_url(owner, repo, branch, path):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def fetch_tree(owner, repo, branch):
    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    r = requests.get(api, timeout=30)
    r.raise_for_status()
    return [p["path"] for p in r.json()["tree"] if p["type"] == "blob"]


def download(owner, repo, branch, paths, dest):
    for rel in paths:
        url = build_url(owner, repo, branch, rel)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)


def ascii_tree_with_links(
    root: pathlib.Path,
    base: pathlib.Path,  # 追加: ツリー全体のルート
    owner,
    repo,
    branch,
    prefix="",
):
    # Skip dotfiles / dot-dirs at display stage as well
    entries = sorted(
        [p for p in root.iterdir() if not p.name.startswith(".")],
        key=lambda p: (p.is_file(), p.name),
    )
    last = len(entries) - 1
    for i, entry in enumerate(entries):
        connector = "└── " if i == last else "├── "
        # ルート(base)からの相対パスを RAW URL に使う
        rel = entry.relative_to(base)
        if entry.is_file():
            raw = build_url(owner, repo, branch, rel.as_posix())
            print(f"{prefix}{connector}{entry.name:<25} {raw}")
        else:
            print(f"{prefix}{connector}{entry.name}/")
            extension = "    " if i == last else "│   "
            ascii_tree_with_links(entry, base, owner, repo, branch, prefix + extension)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_from_repo_links.py <owner/repo> [branch] [dest_dir]")
        sys.exit(1)
    owner_repo = sys.argv[1]
    branch = sys.argv[2] if len(sys.argv) > 2 else "main"
    dest_dir = sys.argv[3] if len(sys.argv) > 3 else owner_repo.split("/")[-1] + "_src"

    owner, repo = owner_repo.split("/")
    # --- exclude dotfiles (leading ".") ---------------------------------
    paths_all = fetch_tree(owner, repo, branch)

    # --- exclude dotfiles AND dot-directories ------------------------------
    def is_dot_component(path_str: str) -> bool:
        """Return True if any component of the path starts with a dot."""
        return any(part.startswith(".") for part in pathlib.Path(path_str).parts)

    paths = [p for p in paths_all if not is_dot_component(p)]

    dest = pathlib.Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    download(owner, repo, branch, paths, dest)

    print("\n# File Tree with RAW URLs\n")
    print(dest.name + "/")
    ascii_tree_with_links(dest, dest, owner, repo, branch)
    skipped = len(paths_all) - len(paths)
    if skipped:
        print(f"\n⚠️  {skipped} dot item(s) (files or dirs) were skipped.")


if __name__ == "__main__":
    main()
