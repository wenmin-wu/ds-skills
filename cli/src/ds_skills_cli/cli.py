"""ds-skills CLI — agent-friendly interface to ds-skills.com.

Exit codes: 0=success, 1=error, 2=not found, 3=invalid input.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from ds_skills_cli.client import ApiError, Client
from ds_skills_cli.output import emit_json, emit_table, log

# Agent install directories (default per platform)
AGENT_DIRS = {
    "claude-code": Path.home() / ".claude" / "skills",
    "cursor": Path.home() / ".cursor" / "rules",
    "codex": Path.home() / ".codex" / "skills",
}

OUTCOMES = ["success", "failed", "thumbup", "thumbdown"]


def _build_parser() -> argparse.ArgumentParser:
    # Shared flags available on every subcommand (before or after subcommand name)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="Output JSON to stdout (agent mode)"
    )

    p = argparse.ArgumentParser(
        prog="ds-skills",
        description="Browse, search, and pull data science skills from ds-skills.com",
        parents=[common],
    )

    sub = p.add_subparsers(dest="command")

    # --- list ---
    ls = sub.add_parser("list", help="List skills", parents=[common])
    ls.add_argument("--domain", "-d", help="Filter by domain")
    ls.add_argument("--page", type=int, default=1)
    ls.add_argument("--limit", type=int, default=200)

    # --- search ---
    sr = sub.add_parser("search", help="Search skills by keyword", parents=[common])
    sr.add_argument("query", help="Search query")
    sr.add_argument("--domain", "-d", help="Filter by domain")
    sr.add_argument("--page", type=int, default=1)
    sr.add_argument("--limit", type=int, default=50)

    # --- show ---
    sh = sub.add_parser("show", help="Show full skill detail (e.g. nlp/deberta-classification)", parents=[common])
    sh.add_argument("skill", help="domain/slug (e.g. nlp/deberta-classification)")

    # --- pull ---
    pl = sub.add_parser("pull", help="Download and extract skills", parents=[common])
    pl.add_argument(
        "target",
        help="domain/slug for one skill, or domain name for all skills in a domain",
    )
    pl.add_argument(
        "--dest", "-o", type=Path, default=Path("."), help="Destination directory"
    )

    # --- install ---
    ins = sub.add_parser("install", help="Install skills to an AI agent's skill directory", parents=[common])
    ins.add_argument("skill", nargs="?", default=None, help="Optional domain/slug to install a single skill (e.g. nlp/deberta-classification)")
    ins.add_argument(
        "--agent",
        "-a",
        required=True,
        choices=list(AGENT_DIRS.keys()),
        help="Target agent",
    )
    ins.add_argument("--domain", "-d", help="Only install skills from this domain")
    ins.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Override default agent directory",
    )

    # --- stats ---
    sub.add_parser("stats", help="Show aggregate statistics", parents=[common])

    # --- feedback ---
    fb = sub.add_parser("feedback", help="Submit feedback for a skill", parents=[common])
    fb.add_argument("skill", help="domain/slug (e.g. nlp/deberta-classification)")
    fb.add_argument(
        "--outcome",
        required=True,
        choices=OUTCOMES,
        help="Outcome: success, failed, thumbup, thumbdown",
    )
    fb.add_argument("--traces", help="Error traces summary (PII removed)")
    fb.add_argument("--username", help="Override configured username")

    # --- config ---
    cfg = sub.add_parser("config", help="Get or set CLI configuration", parents=[common])
    cfg.add_argument("key", nargs="?", help="Config key (e.g. username, hub_url)")
    cfg.add_argument("value", nargs="?", help="Value to set (omit to read)")

    return p


def _parse_skill_ref(ref: str) -> tuple[str, str | None]:
    """Parse 'domain/slug' or 'domain'. Returns (domain, slug_or_None)."""
    parts = ref.strip("/").split("/", 1)
    domain = parts[0]
    slug = parts[1] if len(parts) > 1 else None
    return domain, slug


def cmd_list(client: Client, args: argparse.Namespace) -> int:
    data = client.list_skills(domain=args.domain, page=args.page, limit=args.limit)
    if args.json:
        emit_json(data)
    else:
        skills = data.get("skills", [])
        log(f"{data.get('total', len(skills))} skills")
        emit_table(
            skills,
            ["domain", "slug", "description"],
            [12, 40, 60],
        )
    return 0


def cmd_search(client: Client, args: argparse.Namespace) -> int:
    data = client.search(
        query=args.query, domain=args.domain, page=args.page, limit=args.limit
    )
    if args.json:
        emit_json(data)
    else:
        results = data.get("results", [])
        facets = data.get("facets", {})
        log(f'{data.get("total", len(results))} results for "{args.query}"')
        if facets:
            log("  " + "  ".join(f"{d}:{n}" for d, n in facets.items()))
        emit_table(results, ["domain", "slug", "description"], [12, 40, 60])
    return 0


def cmd_show(client: Client, args: argparse.Namespace) -> int:
    domain, slug = _parse_skill_ref(args.skill)
    if not slug:
        log(f"ERROR: Invalid skill reference '{args.skill}'. Expected domain/slug.")
        return 3
    data = client.show_skill(domain, slug)
    if args.json:
        emit_json(data)
    else:
        print(data.get("content", ""))
    return 0


def cmd_pull(client: Client, args: argparse.Namespace) -> int:
    domain, slug = _parse_skill_ref(args.target)
    dest = args.dest

    if slug:
        log(f"Pulling {domain}/{slug} → {dest}")
        files = client.pull_skill(domain, slug, dest)
    else:
        log(f"Pulling all {domain} skills → {dest}")
        files = client.pull_domain(domain, dest)

    if args.json:
        emit_json({"files": files, "count": len(files)})
    else:
        log(f"{len(files)} files extracted")
        for f in files:
            print(f)
    return 0


def cmd_install(client: Client, args: argparse.Namespace) -> int:
    dest = args.dest or AGENT_DIRS[args.agent]

    # Single skill: ds-skills install nlp/deberta-classification --agent claude-code
    if args.skill:
        parts = args.skill.split("/", 1)
        if len(parts) != 2:
            log(f"Error: skill must be domain/slug (e.g. nlp/deberta-classification), got: {args.skill}", file=sys.stderr)
            return 1
        domain, slug = parts
        log(f"Installing {args.skill} to {dest} (agent: {args.agent})")
        files = client.pull_skill(domain, slug, dest)
        if args.json:
            emit_json({"agent": args.agent, "dest": str(dest), "skill": args.skill, "files": files, "count": len(files)})
        else:
            log(f"Done. {len(files)} files installed to {dest}")
        return 0

    # Domain or all skills
    log(f"Installing skills to {dest} (agent: {args.agent})")

    if args.domain:
        domains = [args.domain]
    else:
        stats = client.stats()
        domains = list(stats.get("domains", {}).keys())

    all_files: list[str] = []
    for domain in domains:
        log(f"  pulling {domain}...")
        files = client.pull_domain(domain, dest)
        all_files.extend(files)

    if args.json:
        emit_json({"agent": args.agent, "dest": str(dest), "files": all_files, "count": len(all_files)})
    else:
        log(f"Done. {len(all_files)} files installed to {dest}")
    return 0


def cmd_stats(client: Client, args: argparse.Namespace) -> int:
    data = client.stats()
    if args.json:
        emit_json(data)
    else:
        print(f"Total skills: {data.get('total_skills', '?')}")
        print(f"Competitions: {data.get('competitions_processed', '?')}")
        for d, n in data.get("domains", {}).items():
            print(f"  {d}: {n}")
    return 0


def cmd_feedback(client: Client, args: argparse.Namespace) -> int:
    from ds_skills_cli import config

    domain, slug = _parse_skill_ref(args.skill)
    if not slug:
        log(f"ERROR: Invalid skill reference '{args.skill}'. Expected domain/slug.")
        return 3

    username = args.username or config.get("username")
    payload = {
        "skill_domain": domain,
        "skill_slug": slug,
        "outcome": args.outcome,
        "cli_version": _get_version(),
        "os": platform.system().lower(),
        "os_version": platform.release(),
        "username": username or None,
    }
    if args.traces:
        payload["traces_summary"] = args.traces

    data = client.post_feedback(payload)
    if args.json:
        emit_json(data)
    else:
        log(f"Feedback submitted: {args.outcome} for {domain}/{slug}")
    return 0


def cmd_config(client: Client, args: argparse.Namespace) -> int:
    from ds_skills_cli import config

    if not args.key:
        # Show all config
        data = config.get_all()
        if args.json:
            emit_json(data)
        else:
            for k, v in data.items():
                print(f"{k}: {v}")
        return 0

    if args.value is not None:
        config.set_value(args.key, args.value)
        if args.json:
            emit_json({"key": args.key, "value": args.value})
        else:
            log(f"Set {args.key} = {args.value}")
    else:
        val = config.get(args.key)
        if args.json:
            emit_json({"key": args.key, "value": val})
        else:
            print(val)
    return 0


def _get_version() -> str:
    try:
        from ds_skills_cli import __version__
        return __version__
    except Exception:
        return "unknown"


_DISPATCH = {
    "list": cmd_list,
    "search": cmd_search,
    "show": cmd_show,
    "pull": cmd_pull,
    "install": cmd_install,
    "stats": cmd_stats,
    "feedback": cmd_feedback,
    "config": cmd_config,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        sys.exit(3)

    # Config command doesn't need a client
    if args.command == "config":
        code = cmd_config(None, args)  # type: ignore[arg-type]
        sys.exit(code)

    from ds_skills_cli import config
    client = Client(base_url=config.get_hub_url())

    try:
        code = _DISPATCH[args.command](client, args)
        sys.exit(code)
    except ApiError as exc:
        if exc.status == 404:
            log(f"ERROR: Not found. {exc.message}")
            if args.json:
                emit_json({"error": exc.message, "status": 404})
            sys.exit(2)
        else:
            log(f"ERROR: {exc}")
            if args.json:
                emit_json({"error": str(exc), "status": exc.status})
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        log(f"ERROR: {exc}")
        if args.json:
            emit_json({"error": str(exc)})
        sys.exit(1)
