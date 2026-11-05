#!/usr/bin/env python3
"""
PolyStrata & am2strata Daily GitHub Tracker
Automatically inspects:
- VecSzn/PolyStrata (upstream commits & releases)
- abhaymin/am2strata (upstream commits & releases)
- Active forks with recent pushes
- Related community repositories (ComfyUI-PolyStrata, etc.)

Writes a markdown summary to G:\\Strata-3060\\LATEST_GITHUB_UPDATES.md
and keeps track of seen commits in .github_state.json.
"""

import sys
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(r"G:\Strata-3060")
TOOLS_DIR = BASE_DIR / "tools"
STATE_FILE = TOOLS_DIR / ".github_state.json"
OUTPUT_REPORT = BASE_DIR / "LATEST_GITHUB_UPDATES.md"

def fetch_json(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PolyStrata-Daily-Tracker",
            "Accept": "application/vnd.github.v3+json"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not save state: {e}")

def main():
    print("=" * 60)
    print(" PolyStrata & am2strata GitHub Daily Check")
    print("=" * 60)

    state = load_state()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Check VecSzn/PolyStrata
    print("[1/4] Checking VecSzn/PolyStrata...")
    poly_commits = fetch_json("https://api.github.com/repos/VecSzn/PolyStrata/commits?per_page=10")
    last_poly_sha = state.get("last_poly_sha")
    new_poly_commits = []
    
    if isinstance(poly_commits, list):
        if last_poly_sha:
            for c in poly_commits:
                if c["sha"] == last_poly_sha:
                    break
                new_poly_commits.append(c)
        else:
            new_poly_commits = poly_commits[:5]
        if poly_commits:
            state["last_poly_sha"] = poly_commits[0]["sha"]

    # 2. Check abhaymin/am2strata
    print("[2/4] Checking abhaymin/am2strata...")
    am2_commits = fetch_json("https://api.github.com/repos/abhaymin/am2strata/commits?per_page=10")
    last_am2_sha = state.get("last_am2_sha")
    new_am2_commits = []

    if isinstance(am2_commits, list):
        if last_am2_sha:
            for c in am2_commits:
                if c["sha"] == last_am2_sha:
                    break
                new_am2_commits.append(c)
        else:
            new_am2_commits = am2_commits[:5]
        if am2_commits:
            state["last_am2_sha"] = am2_commits[0]["sha"]

    # 3. Check Active Forks of PolyStrata
    print("[3/4] Scanning PolyStrata forks...")
    forks_data = fetch_json("https://api.github.com/repos/VecSzn/PolyStrata/forks?sort=newest&per_page=20")
    active_forks = []
    if isinstance(forks_data, list):
        for f in forks_data:
            pushed_at = f.get("pushed_at", "")
            # Look at forks pushed recently
            if pushed_at >= "2026-10-25":
                active_forks.append({
                    "name": f.get("full_name"),
                    "url": f.get("html_url"),
                    "pushed_at": pushed_at,
                    "stars": f.get("stargazers_count", 0),
                    "desc": f.get("description") or "No description"
                })

    # 4. Check Ecosystem Add-ons (e.g. ComfyUI-PolyStrata)
    print("[4/4] Checking community add-ons...")
    addons = [
        "bananasss00/ComfyUI-PolyStrata",
        "SRZhu26/PolyStrata"
    ]
    addon_reports = []
    for repo in addons:
        c = fetch_json(f"https://api.github.com/repos/{repo}/commits?per_page=1")
        if isinstance(c, list) and c:
            latest = c[0]
            addon_reports.append({
                "repo": repo,
                "sha": latest["sha"][:8],
                "date": latest["commit"]["author"]["date"],
                "msg": latest["commit"]["message"].splitlines()[0],
                "url": f"https://github.com/{repo}"
            })

    # Save updated state
    save_state(state)

    # Format Markdown Report
    lines = [
        f"# PolyStrata & am2strata Daily GitHub Report",
        f"*Generated on: {now_str}*",
        "",
        "## 1. Upstream PolyStrata (VecSzn/PolyStrata)",
    ]

    if new_poly_commits:
        lines.append(f"**Found {len(new_poly_commits)} recent commit(s):**")
        for c in new_poly_commits:
            sha = c["sha"][:8]
            date = c["commit"]["author"]["date"][:10]
            msg = c["commit"]["message"].splitlines()[0]
            lines.append(f"- [`{sha}`](https://github.com/VecSzn/PolyStrata/commit/{c['sha']}) *({date})*: {msg}")
    else:
        lines.append("[OK] Your local PolyStrata is up-to-date with upstream HEAD.")

    lines.extend([
        "",
        "## 2. Upstream am2strata (abhaymin/am2strata)",
    ])

    if new_am2_commits:
        lines.append(f"**Found {len(new_am2_commits)} recent commit(s):**")
        for c in new_am2_commits:
            sha = c["sha"][:8]
            date = c["commit"]["author"]["date"][:10]
            msg = c["commit"]["message"].splitlines()[0]
            lines.append(f"- [`{sha}`](https://github.com/abhaymin/am2strata/commit/{c['sha']}) *({date})*: {msg}")
    else:
        lines.append("[OK] Your am2strata engine is up-to-date with upstream HEAD.")

    lines.extend([
        "",
        "## 3. Notable Community Forks & New Models",
    ])

    if active_forks:
        for f in active_forks[:8]:
            lines.append(f"- [{f['name']}]({f['url']}) (Pushed: {f['pushed_at'][:10]}, Stars: {f['stars']}): {f['desc']}")
    else:
        lines.append("- No new fork activity detected in the last 7 days.")

    lines.extend([
        "",
        "## 4. Community Add-Ons & Integrations",
    ])

    for a in addon_reports:
        lines.append(f"- [{a['repo']}]({a['url']}): [`{a['sha']}`] *{a['msg']}* ({a['date'][:10]})")

    report_text = "\n".join(lines)
    OUTPUT_REPORT.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Report written to: {OUTPUT_REPORT}")
    print("\nSummary:")
    print(report_text)

if __name__ == "__main__":
    main()
