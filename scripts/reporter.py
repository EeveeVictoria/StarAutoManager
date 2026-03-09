"""Reporting and notification system for StarAutoManager."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

from .models import Confidence, RunReport

logger = logging.getLogger(__name__)


class Reporter:
    """Generate reports and send notifications."""

    def __init__(self, config: dict, github_token: str):
        self.config = config
        self.github_token = github_token
        self.notify_config = config.get("notifications", {})

    # ------------------------------------------------------------------
    # GitHub Issue report
    # ------------------------------------------------------------------

    def create_issue_report(self, report: RunReport) -> Optional[str]:
        """Create a GitHub Issue with the run report."""
        if not self.notify_config.get("issue", True):
            return None

        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if not repo:
            logger.warning("GITHUB_REPOSITORY not set — skipping issue creation")
            return None

        title = self._build_issue_title(report)
        body = self._build_issue_body(report)
        labels = []
        label = self.notify_config.get("issue_label", "star-manager")
        if label:
            labels.append(label)

        try:
            url = f"https://api.github.com/repos/{repo}/issues"
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"title": title, "body": body, "labels": labels},
                timeout=30,
            )
            resp.raise_for_status()
            issue_url = resp.json().get("html_url", "")
            logger.info("Created issue: %s", issue_url)
            return issue_url
        except Exception as exc:
            logger.error("Failed to create issue: %s", exc)
            return None

    def _build_issue_title(self, report: RunReport) -> str:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        mode = "🔍 Dry Run" if report.dry_run else "✅ Applied"
        return f"[StarAutoManager] {mode} — {len(report.categorizations)} repos categorized ({date_str})"

    def _build_issue_body(self, report: RunReport) -> str:
        lines: list[str] = []

        lines.append("# ⭐ StarAutoManager Report\n")
        lines.append(f"**Run time:** {report.timestamp}")
        lines.append(
            f"**Mode:** {'🔍 Dry Run (no changes applied)' if report.dry_run else '✅ Changes Applied'}\n"
        )

        # Overview
        lines.append("## 📊 Overview\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total starred repos | {report.total_starred} |")
        lines.append(f"| Existing lists | {report.total_lists} |")
        lines.append(f"| Uncategorized repos | {report.total_uncategorized} |")
        lines.append(f"| Categorized this run | {len(report.categorizations)} |")
        lines.append(f"| High confidence | {report.high_confidence_count} |")
        lines.append(f"| Low confidence | {report.low_confidence_count} |")
        lines.append(f"| New lists created | {len(report.new_lists_created)} |")
        lines.append("")

        # Categorization details
        if report.categorizations:
            lines.append("## 📁 Categorization Results\n")
            # Group by list
            by_list: dict[str, list] = {}
            for cat in report.categorizations:
                by_list.setdefault(cat.list_name, []).append(cat)

            for list_name, cats in sorted(by_list.items()):
                new_tag = " 🆕" if any(c.is_new_list for c in cats) else ""
                lines.append(f"### {list_name}{new_tag}\n")
                for cat in cats:
                    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                        cat.confidence.value, "⚪"
                    )
                    lines.append(
                        f"- {conf_icon} **{cat.repo_full_name}** — {cat.reason}"
                    )
                lines.append("")

        # New lists
        if report.new_lists_created:
            lines.append("## 🆕 New Lists Created\n")
            for name in report.new_lists_created:
                lines.append(f"- {name}")
            lines.append("")

        # Language stats
        if report.language_stats:
            lines.append("## 🌐 Language Distribution (Top 15)\n")
            lines.append("| Language | Count |")
            lines.append("|----------|-------|")
            for lang, count in list(report.language_stats.items())[:15]:
                lines.append(f"| {lang} | {count} |")
            lines.append("")

        # Topic cloud
        if report.topic_stats:
            lines.append("## 🏷️ Top Topics\n")
            top_topics = list(report.topic_stats.items())[:20]
            lines.append(
                " ".join(f"`{topic}` ({count})" for topic, count in top_topics)
            )
            lines.append("")

        # List health
        if report.list_health:
            lines.append("## 🏥 List Health Check\n")
            for name, warning in report.list_health.items():
                lines.append(f"- **{name}**: {warning}")
            lines.append("")

        # Stale repos
        if report.stale_repos:
            lines.append("## 🗑️ Potentially Stale Repos\n")
            lines.append(
                "<details><summary>Click to expand (%d repos)</summary>\n"
                % len(report.stale_repos)
            )
            for stale in report.stale_repos:
                lines.append(stale.summary())
            lines.append("\n</details>\n")

        # Duplicate groups
        if report.duplicate_groups:
            lines.append("## 🔄 Potentially Duplicate Repos\n")
            for group in report.duplicate_groups:
                lines.append(f"**{group.description}:**")
                for repo in group.repos:
                    lines.append(f"- {repo.full_name}")
                lines.append("")

        # Errors
        if report.errors:
            lines.append("## ❌ Errors\n")
            for err in report.errors:
                lines.append(f"- `{err}`")
            lines.append("")

        lines.append("---")
        lines.append(
            "*Generated by [StarAutoManager](https://github.com/your-user/StarAutoManager)*"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # STARS.md generation
    # ------------------------------------------------------------------

    def generate_stars_md(
        self,
        report: RunReport,
        all_lists: list,
        output_path: str = "STARS.md",
    ) -> None:
        """Generate a beautiful STARS.md file with categorized stars."""
        if not self.notify_config.get("summary_in_readme", True):
            return

        lines: list[str] = []
        lines.append("# ⭐ My Starred Repositories\n")
        lines.append(
            f"> Auto-organized by [StarAutoManager](https://github.com/your-user/StarAutoManager) "
            f"| Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        )

        # Stats
        lines.append(
            f"**{report.total_starred}** repos across **{report.total_lists}** lists\n"
        )

        # TOC
        if all_lists:
            lines.append("## 📑 Table of Contents\n")
            for sl in all_lists:
                anchor = sl.name.lower().replace(" ", "-").replace("/", "")
                lines.append(f"- [{sl.name}](#{anchor}) ({sl.repo_count})")
            lines.append("")

        # List sections
        for sl in all_lists:
            lines.append(f"## {sl.name}\n")
            if sl.description:
                lines.append(f"*{sl.description}*\n")

            lines.append("| Repository | Description | Language | Stars |")
            lines.append("|------------|-------------|----------|-------|")
            for repo in sorted(sl.repos, key=lambda r: r.stargazer_count, reverse=True):
                desc = (
                    (repo.description[:80] + "...")
                    if len(repo.description) > 80
                    else repo.description
                )
                desc = desc.replace("|", "\\|")
                lines.append(
                    f"| [{repo.full_name}]({repo.url}) | {desc} "
                    f"| {repo.language or '-'} | {repo.stargazer_count:,} |"
                )
            lines.append("")

        # Language chart (text-based)
        if report.language_stats:
            lines.append("## 🌐 Language Distribution\n")
            total = sum(report.language_stats.values())
            for lang, count in list(report.language_stats.items())[:10]:
                pct = count / total * 100
                bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                lines.append(f"| {lang:<15} | {bar} | {pct:.1f}% ({count}) |")
            lines.append("")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info("Generated %s", output_path)
        except Exception as exc:
            logger.error("Failed to write %s: %s", output_path, exc)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(report: RunReport) -> None:
        """Print a concise summary to console."""
        logger.info("=" * 60)
        logger.info("STARAUTOMANAGER RUN COMPLETE")
        logger.info("=" * 60)
        logger.info("Mode:           %s", "DRY RUN" if report.dry_run else "APPLIED")
        logger.info("Total starred:  %d", report.total_starred)
        logger.info("Existing lists: %d", report.total_lists)
        logger.info("Uncategorized:  %d", report.total_uncategorized)
        logger.info("Categorized:    %d", len(report.categorizations))
        logger.info("  High conf:    %d", report.high_confidence_count)
        logger.info("  Low conf:     %d", report.low_confidence_count)
        logger.info("New lists:      %d", len(report.new_lists_created))
        logger.info("Stale repos:    %d", len(report.stale_repos))
        logger.info("Duplicates:     %d groups", len(report.duplicate_groups))
        if report.errors:
            logger.error("Errors:         %d", len(report.errors))
            for err in report.errors:
                logger.error("  • %s", err)
