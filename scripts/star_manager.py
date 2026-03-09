"""Star Manager — orchestration logic for the learn-then-classify pipeline."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Optional

from .github_client import GitHubGraphQLClient
from .llm_client import LLMClient
from .models import (
    Cache,
    Categorization,
    Confidence,
    DuplicateGroup,
    Repository,
    RunReport,
    StaleRepo,
    StarList,
)

logger = logging.getLogger(__name__)


class StarManager:
    """Orchestrates the full star categorization pipeline."""

    def __init__(
        self,
        github: GitHubGraphQLClient,
        llm: LLMClient,
        config: dict,
    ):
        self.github = github
        self.llm = llm
        self.config = config
        self.cat_config = config.get("categorization", {})
        self.adv_config = config.get("advanced", {})
        self.cache = Cache.load(self.adv_config.get("cache_file", ".star-cache.json"))

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> RunReport:
        """Execute the full categorization pipeline."""
        report = RunReport(
            timestamp=datetime.utcnow().isoformat(),
            dry_run=self.cat_config.get("dry_run", False),
        )

        try:
            # Step 1: Authenticate & detect username
            username = self._resolve_username()

            # Step 2: Fetch existing Star Lists (training data)
            logger.info("=" * 60)
            logger.info(
                "STEP 1: Fetching existing Star Lists (learning user habits)..."
            )
            existing_lists = self.github.get_user_lists(username)
            report.total_lists = len(existing_lists)
            self._log_lists_summary(existing_lists)

            # Step 3: Fetch all starred repos
            logger.info("=" * 60)
            logger.info("STEP 2: Fetching all starred repositories...")
            max_repos = self.cat_config.get("max_repos_per_run", 0)
            all_starred = self.github.get_starred_repos(max_repos=max_repos)
            report.total_starred = len(all_starred)

            # Step 4: Identify uncategorized repos
            logger.info("=" * 60)
            logger.info("STEP 3: Identifying uncategorized repos...")
            uncategorized = self._find_uncategorized(all_starred, existing_lists)
            report.total_uncategorized = len(uncategorized)
            logger.info(
                "Found %d uncategorized repos out of %d total",
                len(uncategorized),
                len(all_starred),
            )

            # Step 4b: Filter by config
            uncategorized = self._apply_filters(uncategorized)

            # Step 4c: Remove cached repos (already categorized in previous runs)
            if not self.config.get("force_recategorize", False):
                uncategorized = [
                    r for r in uncategorized if not self.cache.is_cached(r.full_name)
                ]
                logger.info("%d repos remaining after cache filter", len(uncategorized))

            # Step 5: LLM categorization (learn then classify)
            if uncategorized:
                logger.info("=" * 60)
                logger.info("STEP 4: LLM categorization (learn → classify)...")
                categorizations = self.llm.categorize_repos(
                    uncategorized, existing_lists
                )
                report.categorizations = categorizations

                # Second pass: fetch README for low-confidence repos
                if self.cat_config.get("fetch_readme", True):
                    low_conf = [
                        c for c in categorizations if c.confidence == Confidence.LOW
                    ]
                    if low_conf:
                        logger.info(
                            "Fetching README for %d low-confidence repos...",
                            len(low_conf),
                        )
                        repos_with_readme = self._enrich_with_readme(
                            low_conf, uncategorized
                        )
                        if repos_with_readme:
                            second_results = self.llm.categorize_repos(
                                repos_with_readme, existing_lists
                            )
                            self._merge_second_pass(
                                report.categorizations, second_results
                            )

                # Step 6: Apply changes
                if not report.dry_run:
                    logger.info("=" * 60)
                    logger.info("STEP 5: Applying categorization changes...")
                    self._apply_categorizations(report.categorizations, existing_lists)
                else:
                    logger.info("DRY RUN — no changes applied")
            else:
                logger.info("No uncategorized repos to process. All clean!")

            # Step 7: Smart analysis features
            logger.info("=" * 60)
            logger.info("STEP 6: Running smart analysis...")

            # Language & topic stats
            report.language_stats = self._compute_language_stats(all_starred)
            report.topic_stats = self._compute_topic_stats(all_starred)

            # List health check
            report.list_health = self._check_list_health(existing_lists)

            # Stale repo detection
            report.stale_repos = self._detect_stale(all_starred)

            # Duplicate detection
            report.duplicate_groups = self._detect_duplicates(all_starred)

            # Step 8: Save cache
            self._update_cache(report.categorizations)
            self.cache.last_run = report.timestamp
            self.cache.save(self.adv_config.get("cache_file", ".star-cache.json"))
            logger.info("Cache saved.")

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            report.errors.append(str(exc))

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_username(self) -> str:
        username = self.config.get("github", {}).get("username", "")
        if not username:
            username = self.github.get_viewer_login()
        logger.info("Operating as user: %s", username)
        return username

    def _find_uncategorized(
        self,
        all_starred: list[Repository],
        existing_lists: list[StarList],
    ) -> list[Repository]:
        """Find repos that are starred but not in any list."""
        categorized_ids: set[str] = set()
        # Build map of repo_id → list_ids for later use
        repo_list_map: dict[str, list[str]] = {}

        for sl in existing_lists:
            for repo in sl.repos:
                categorized_ids.add(repo.node_id)
                repo_list_map.setdefault(repo.node_id, []).append(sl.node_id)

        # Annotate starred repos with their list memberships
        for repo in all_starred:
            repo.list_ids = repo_list_map.get(repo.node_id, [])

        return [r for r in all_starred if r.node_id not in categorized_ids]

    def _apply_filters(self, repos: list[Repository]) -> list[Repository]:
        """Apply config-based filters."""
        filtered = repos
        if self.cat_config.get("ignore_archived", True):
            before = len(filtered)
            filtered = [r for r in filtered if not r.is_archived]
            skipped = before - len(filtered)
            if skipped:
                logger.info("Filtered %d archived repos", skipped)

        if self.cat_config.get("ignore_forks", False):
            # We don't have is_fork from the query, but fork_count can hint
            pass

        max_per_run = self.cat_config.get("max_repos_per_run", 0)
        if max_per_run and len(filtered) > max_per_run:
            logger.info("Limiting to %d repos per run", max_per_run)
            filtered = filtered[:max_per_run]

        return filtered

    def _enrich_with_readme(
        self,
        low_conf: list[Categorization],
        repos: list[Repository],
    ) -> list[Repository]:
        """Fetch README snippets for low-confidence repos."""
        repo_map = {r.full_name: r for r in repos}
        enriched = []

        for cat in low_conf[:20]:  # Limit README fetches
            repo = repo_map.get(cat.repo_full_name)
            if not repo:
                continue
            parts = repo.full_name.split("/", 1)
            if len(parts) != 2:
                continue
            readme = self.github.fetch_readme(parts[0], parts[1])
            if readme:
                repo.readme_snippet = readme
                enriched.append(repo)

        logger.info("Fetched README for %d repos", len(enriched))
        return enriched

    @staticmethod
    def _merge_second_pass(
        original: list[Categorization],
        second_pass: list[Categorization],
    ) -> None:
        """Replace low-confidence results with second-pass results in place."""
        second_map = {c.repo_full_name: c for c in second_pass}
        for i, cat in enumerate(original):
            if cat.repo_full_name in second_map:
                original[i] = second_map[cat.repo_full_name]

    # ------------------------------------------------------------------
    # Apply changes
    # ------------------------------------------------------------------

    def _apply_categorizations(
        self,
        categorizations: list[Categorization],
        existing_lists: list[StarList],
    ) -> None:
        """Apply categorization decisions via GitHub GraphQL mutations."""
        min_conf = Confidence(self.cat_config.get("min_confidence", "medium"))

        # Map list names to IDs
        list_name_to_id: dict[str, str] = {sl.name: sl.node_id for sl in existing_lists}
        new_lists_created: list[str] = []

        for cat in categorizations:
            if cat.confidence < min_conf:
                logger.info(
                    "Skipping %s (confidence=%s < min=%s)",
                    cat.repo_full_name,
                    cat.confidence.value,
                    min_conf.value,
                )
                continue

            # Create new list if needed
            if cat.is_new_list and cat.list_name not in list_name_to_id:
                if len(list_name_to_id) >= 32:
                    logger.warning(
                        "Cannot create list '%s' — 32-list limit reached",
                        cat.list_name,
                    )
                    continue

                try:
                    new_list = self.github.create_list(
                        cat.list_name, cat.new_list_description
                    )
                    list_name_to_id[new_list.name] = new_list.node_id
                    new_lists_created.append(new_list.name)
                    logger.info("Created new list: %s", new_list.name)
                except Exception as exc:
                    logger.error("Failed to create list '%s': %s", cat.list_name, exc)
                    continue

            target_list_id = list_name_to_id.get(cat.list_name)
            if not target_list_id:
                logger.warning(
                    "List '%s' not found for repo %s — skipping",
                    cat.list_name,
                    cat.repo_full_name,
                )
                continue

            # CRITICAL: updateUserListsForItem REPLACES all memberships
            # We must preserve existing list memberships + add new one
            # repo.list_ids was populated in _find_uncategorized
            current_list_ids: list[str] = []  # uncategorized repos have no list_ids
            desired_list_ids = list(set(current_list_ids + [target_list_id]))

            try:
                self.github.update_repo_lists(cat.repo_node_id, desired_list_ids)
                logger.info(
                    "Assigned %s → '%s' (confidence=%s)",
                    cat.repo_full_name,
                    cat.list_name,
                    cat.confidence.value,
                )
            except Exception as exc:
                logger.error(
                    "Failed to assign %s → '%s': %s",
                    cat.repo_full_name,
                    cat.list_name,
                    exc,
                )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_language_stats(repos: list[Repository]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for r in repos:
            lang = r.language or "Unknown"
            counter[lang] += 1
        return dict(counter.most_common(30))

    @staticmethod
    def _compute_topic_stats(repos: list[Repository]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for r in repos:
            for t in r.topics:
                counter[t] += 1
        return dict(counter.most_common(40))

    @staticmethod
    def _check_list_health(lists: list[StarList]) -> dict[str, str]:
        warnings: dict[str, str] = {}
        for sl in lists:
            if sl.repo_count > 50:
                warnings[sl.name] = (
                    f"⚠️ Too many repos ({sl.repo_count}) — consider splitting"
                )
            elif sl.repo_count < 3:
                warnings[sl.name] = f"⚠️ Only {sl.repo_count} repos — consider merging"
        return warnings

    def _detect_stale(self, repos: list[Repository]) -> list[StaleRepo]:
        """Detect stale repos — first heuristic check, then LLM refinement."""
        stale: list[StaleRepo] = []

        for r in repos:
            reasons: list[str] = []
            if r.is_archived:
                reasons.append("Repository is archived")
            if r.days_since_pushed and r.days_since_pushed > 730:
                reasons.append(f"No pushes in {r.days_since_pushed} days")
            if (
                r.days_since_starred
                and r.days_since_starred > 365
                and r.stargazer_count < 50
            ):
                reasons.append("Starred over a year ago with low star count")
            if reasons:
                stale.append(StaleRepo(repo=r, reasons=reasons))

        if stale:
            logger.info("Detected %d potentially stale repos", len(stale))

        return stale[:30]  # Cap for report readability

    def _detect_duplicates(self, repos: list[Repository]) -> list[DuplicateGroup]:
        """Detect potentially duplicate/overlapping repos via LLM."""
        if len(repos) < 10:
            return []

        try:
            results = self.llm.detect_duplicates(repos)
            groups = []
            repo_map = {r.full_name: r for r in repos}
            for item in results:
                group_repos = [
                    repo_map[name] for name in item.get("repos", []) if name in repo_map
                ]
                if len(group_repos) >= 2:
                    groups.append(
                        DuplicateGroup(
                            description=item.get("description", ""),
                            repos=group_repos,
                        )
                    )
            return groups
        except Exception as exc:
            logger.warning("Duplicate detection failed: %s", exc)
            return []

    def _update_cache(self, categorizations: list[Categorization]) -> None:
        for cat in categorizations:
            self.cache.add(cat)

    def _log_lists_summary(self, lists: list[StarList]) -> None:
        if not lists:
            logger.info("No existing Star Lists found (cold start mode)")
            return
        logger.info("Found %d existing Star Lists:", len(lists))
        for sl in lists:
            logger.info("  • %s (%d repos)", sl.name, sl.repo_count)
