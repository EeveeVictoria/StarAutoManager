"""StarAutoManager — entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

from .github_client import GitHubGraphQLClient
from .llm_client import LLMClient
from .reporter import Reporter
from .star_manager import StarManager

logger = logging.getLogger("star_manager")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _resolve_config_path(config_path: str) -> Path | None:
    path = Path(config_path)
    if path.exists():
        return path
    for fallback in ("config.example.en.yaml", "config.example.zh.yaml"):
        fb = Path(fallback)
        if fb.exists():
            return fb
    return None


def load_config(config_path: str) -> dict:
    config: dict = {}
    resolved = _resolve_config_path(config_path)
    if resolved:
        with open(resolved, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", resolved)
    else:
        logger.info(
            "No config file found at %s — using defaults + env vars", config_path
        )

    # Ensure nested dicts
    config.setdefault("github", {})
    config.setdefault("llm", {})
    config.setdefault("categorization", {})
    config.setdefault("notifications", {})
    config.setdefault("advanced", {})

    # Environment variable overrides (secrets)
    env_overrides = {
        "GITHUB_TOKEN": ("github", "token"),
        "LLM_BASE_URL": ("llm", "base_url"),
        "LLM_API_KEY": ("llm", "api_key"),
        "LLM_MODEL": ("llm", "model"),
    }
    for env_key, (section, key) in env_overrides.items():
        val = os.environ.get(env_key)
        if val:
            config[section][key] = val

    # CLI / workflow_dispatch overrides
    if os.environ.get("INPUT_DRY_RUN", "").lower() == "true":
        config["categorization"]["dry_run"] = True
    if os.environ.get("INPUT_MAX_REPOS"):
        try:
            config["categorization"]["max_repos_per_run"] = int(
                os.environ["INPUT_MAX_REPOS"]
            )
        except ValueError:
            pass
    if os.environ.get("INPUT_FORCE_RECATEGORIZE", "").lower() == "true":
        config["force_recategorize"] = True

    return config


def validate_config(config: dict) -> None:
    """Validate required config values are present."""
    github_token = config.get("github", {}).get("token")
    if not github_token:
        logger.error("GITHUB_TOKEN is required. Set it as a secret or in config.")
        sys.exit(1)

    llm = config.get("llm", {})
    if not llm.get("base_url"):
        logger.error("LLM_BASE_URL is required. Set it as a GitHub Secret.")
        sys.exit(1)
    if not llm.get("api_key"):
        logger.error("LLM_API_KEY is required. Set it as a GitHub Secret.")
        sys.exit(1)
    if not llm.get("model"):
        logger.error(
            "LLM model is required. Set it in config.yaml or as LLM_MODEL secret."
        )
        sys.exit(1)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="StarAutoManager")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    # Load & validate config
    config = load_config(args.config)
    if args.dry_run:
        config["categorization"]["dry_run"] = True
    validate_config(config)

    github_token = config["github"]["token"]
    llm_config = config["llm"]
    cat_config = config.get("categorization", {})
    adv_config = config.get("advanced", {})

    # Initialize clients
    github = GitHubGraphQLClient(
        token=github_token,
        rate_limit_buffer=adv_config.get("rate_limit_buffer", 500),
    )

    llm = LLMClient(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        temperature=llm_config.get("temperature", 0.3),
        max_tokens=llm_config.get("max_tokens", 4096),
        batch_size=llm_config.get("batch_size", 20),
        max_new_lists=cat_config.get("max_new_lists", 5),
        concurrent_calls=adv_config.get("concurrent_llm_calls", 3),
        language=llm_config.get("language", "en"),
    )

    reporter = Reporter(config=config, github_token=github_token)

    # Run pipeline
    manager = StarManager(github=github, llm=llm, config=config)
    report = manager.run()

    # Output
    reporter.print_summary(report)

    # Refresh lists for STARS.md (if changes were applied)
    if not report.dry_run and report.categorizations:
        try:
            username = (
                config.get("github", {}).get("username") or github.get_viewer_login()
            )
            updated_lists = github.get_user_lists(username)
            reporter.generate_stars_md(report, updated_lists)
        except Exception as exc:
            logger.error("Failed to refresh lists for STARS.md: %s", exc)
    else:
        # Generate with whatever we have
        try:
            username = (
                config.get("github", {}).get("username") or github.get_viewer_login()
            )
            existing_lists = github.get_user_lists(username)
            reporter.generate_stars_md(report, existing_lists)
        except Exception as exc:
            logger.error("Failed to generate STARS.md: %s", exc)

    # Create GitHub issue
    issue_url = reporter.create_issue_report(report)
    if issue_url:
        logger.info("Report issue: %s", issue_url)

    # Exit code
    if report.errors:
        logger.error("Run completed with %d errors", len(report.errors))
        sys.exit(1)

    logger.info("Done! 🎉")


if __name__ == "__main__":
    main()
