"""LLM client for star categorization with learn-then-classify pattern."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from openai import OpenAI  # type: ignore[import-untyped]

from .models import Categorization, Confidence, Repository, StarList

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LEARN = """\
You are an expert at analyzing GitHub repository categorization patterns.

A user has organized some of their starred GitHub repositories into named lists (categories). \
Your job is to study these existing categorized repos carefully, understand the user's \
categorization philosophy, and then classify uncategorized repos following the SAME style.

## User's Existing Star Lists:

{existing_lists}

## Your Analysis Task:

Based on the lists above, you should understand:
1. How granular the user's categories are (broad vs. narrow)
2. Whether the user categorizes by domain (AI, Web, DevOps...), by technology (Python, Rust...), \
by purpose (tools, libraries, learning...), or a combination
3. The user's naming conventions for lists
4. Any patterns in what types of repos go into which list

## Rules:
1. ALWAYS prefer assigning to an existing list when it's a reasonable fit
2. Only suggest a NEW list when a repo truly doesn't fit any existing list
3. Keep total lists under 32 (GitHub's hard limit). Currently: {list_count} lists
4. Maximum {max_new_lists} new lists can be created in this run
5. Each repo must be assigned to exactly ONE list
6. Provide a brief reason for each assignment
7. Rate your confidence: "high" (obvious fit), "medium" (reasonable), "low" (uncertain)
8. Respond ONLY with valid JSON. No markdown fences, no explanation outside JSON.

## Response Format:
[
  {{
    "repo": "owner/repo-name",
    "list": "List Name",
    "reason": "Brief explanation",
    "confidence": "high|medium|low",
    "new_list": false,
    "new_list_description": ""
  }}
]

If suggesting a new list, set "new_list": true and provide "new_list_description".
"""

SYSTEM_PROMPT_LEARN_ZH = """\
你是一位 GitHub 仓库分类分析专家。

一位用户已经将部分 Star 仓库整理到了命名列表（分类）中。\
你的任务是仔细研究这些已分类的仓库，理解用户的分类哲学，然后用相同的风格对未分类仓库进行归类。

## 用户现有的 Star Lists：

{existing_lists}

## 分析要求：

根据以上列表，你需要理解：
1. 用户的分类粒度（粗略 vs 精细）
2. 用户按什么维度分类：领域（AI、Web、DevOps…）、技术（Python、Rust…）、用途（工具、库、学习…）还是混合
3. 用户的列表命名习惯
4. 什么类型的仓库会被归入哪个列表

## 规则：
1. 优先分配到已有列表，只要合理就用现有列表
2. 只有当仓库确实不适合任何现有列表时才建议创建新列表
3. 列表总数不超过 32（GitHub 硬限制），当前：{list_count} 个列表
4. 本次运行最多创建 {max_new_lists} 个新列表
5. 每个仓库必须且只能分配到一个列表
6. 为每个分配提供简短理由
7. 标注置信度："high"（明显匹配）、"medium"（合理）、"low"（不确定）
8. 只返回有效 JSON，不要 markdown 代码块，不要 JSON 之外的文字

## 返回格式：
[
  {{
    "repo": "owner/repo-name",
    "list": "列表名称",
    "reason": "简短理由",
    "confidence": "high|medium|low",
    "new_list": false,
    "new_list_description": ""
  }}
]

如果建议创建新列表，设置 "new_list": true 并填写 "new_list_description"。
"""

SYSTEM_PROMPT_COLD_START = """\
You are an expert at organizing GitHub repositories into meaningful categories.

A user has {total_stars} starred repositories but has NOT created any Star Lists yet. \
Your job is to analyze the repos and propose a sensible categorization structure from scratch.

## Rules:
1. Create broad, meaningful categories (aim for 8-15 lists)
2. Category names should be concise (2-4 words)
3. Think about what a developer would naturally group together
4. Keep total lists under 32 (GitHub's hard limit)
5. Maximum {max_new_lists} new lists can be created in this run
6. Each repo must be assigned to exactly ONE list
7. Consider: domain (AI, Web, DevOps), purpose (tools, libs, learning), language ecosystems
8. Rate your confidence: "high" (obvious fit), "medium" (reasonable), "low" (uncertain)
9. Respond ONLY with valid JSON. No markdown fences, no explanation outside JSON.

## Response Format:
[
  {{
    "repo": "owner/repo-name",
    "list": "List Name",
    "reason": "Brief explanation",
    "confidence": "high|medium|low",
    "new_list": true,
    "new_list_description": "What this list is about"
  }}
]
"""

SYSTEM_PROMPT_COLD_START_ZH = """\
你是一位 GitHub 仓库分类整理专家。

一位用户有 {total_stars} 个 Star 仓库，但尚未创建任何 Star List。\
你需要分析这些仓库，从零开始提出一套合理的分类方案。

## 规则：
1. 创建有意义的大类（目标 8-15 个列表）
2. 分类名称简洁（2-4 个词）
3. 从开发者的角度自然地分组
4. 列表总数不超过 32（GitHub 硬限制）
5. 本次运行最多创建 {max_new_lists} 个新列表
6. 每个仓库必须且只能分配到一个列表
7. 考虑维度：领域（AI、Web、DevOps）、用途（工具、库、学习）、语言生态
8. 标注置信度："high"（明显匹配）、"medium"（合理）、"low"（不确定）
9. 只返回有效 JSON，不要 markdown 代码块，不要 JSON 之外的文字

## 返回格式：
[
  {{
    "repo": "owner/repo-name",
    "list": "列表名称",
    "reason": "简短理由",
    "confidence": "high|medium|low",
    "new_list": true,
    "new_list_description": "这个列表的主题描述"
  }}
]
"""

USER_PROMPT_BATCH = """\
Please categorize the following {count} uncategorized repositories:

{repo_summaries}
"""

USER_PROMPT_BATCH_ZH = """\
请对以下 {count} 个未分类仓库进行归类：

{repo_summaries}
"""

USER_PROMPT_SECOND_PASS = """\
These repos had LOW confidence in the first pass. I've now included their README \
snippets for better context. Please re-categorize them:

{repo_summaries}
"""

USER_PROMPT_SECOND_PASS_ZH = """\
以下仓库在首轮分类中置信度较低，现已附上它们的 README 摘要以提供更多上下文。请重新分类：

{repo_summaries}
"""

PROMPT_STALE_DETECTION = """\
Analyze these repositories and identify ones that may be STALE or no longer useful \
to keep starred. Consider: archived status, years since last push, abandoned projects.

Repos:
{repo_summaries}

Return JSON array:
[
  {{
    "repo": "owner/repo-name",
    "reasons": ["archived", "no updates in 2+ years", ...],
    "suggestion": "unstar|keep|review"
  }}
]
"""

PROMPT_STALE_DETECTION_ZH = """\
分析以下仓库，找出可能已过时或不再值得保留 Star 的项目。\
考虑因素：是否已归档、距上次推送的年数、是否已被放弃。

仓库列表：
{repo_summaries}

返回 JSON 数组：
[
  {{
    "repo": "owner/repo-name",
    "reasons": ["已归档", "超过2年未更新", ...],
    "suggestion": "unstar|keep|review"
  }}
]
"""

PROMPT_DUPLICATE_DETECTION = """\
Analyze these repositories and identify groups that serve SIMILAR purposes \
(e.g., two markdown editors, two CI tools, etc.). Only flag truly overlapping tools.

Repos:
{repo_summaries}

Return JSON array:
[
  {{
    "description": "What these repos have in common",
    "repos": ["owner/repo1", "owner/repo2"]
  }}
]
"""

PROMPT_DUPLICATE_DETECTION_ZH = """\
分析以下仓库，找出功能相似的仓库组（例如两个 Markdown 编辑器、两个 CI 工具等）。\
只标记真正功能重叠的项目。

仓库列表：
{repo_summaries}

返回 JSON 数组：
[
  {{
    "description": "这些仓库的共同点",
    "repos": ["owner/repo1", "owner/repo2"]
  }}
]
"""

PROMPT_RECATEGORIZE_CHECK = """\
Review these ALREADY-CATEGORIZED repos and check if any seem miscategorized \
based on the user's categorization patterns.

Current categorizations:
{categorized_summaries}

User's list definitions:
{list_definitions}

Return JSON array of repos that seem miscategorized:
[
  {{
    "repo": "owner/repo-name",
    "current_list": "Current List Name",
    "suggested_list": "Better List Name",
    "reason": "Why this is a better fit"
  }}
]

If all are well-categorized, return an empty array: []
"""

PROMPT_RECATEGORIZE_CHECK_ZH = """\
审查以下已分类的仓库，根据用户的分类习惯检查是否有分类不当的情况。

当前分类：
{categorized_summaries}

用户的列表定义：
{list_definitions}

返回分类不当的仓库 JSON 数组：
[
  {{
    "repo": "owner/repo-name",
    "current_list": "当前列表名",
    "suggested_list": "建议列表名",
    "reason": "为什么这个分类更合适"
  }}
]

如果所有分类都正确，返回空数组：[]
"""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI-compatible LLM client for repository categorization."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        batch_size: int = 20,
        max_new_lists: int = 5,
        concurrent_calls: int = 3,
        language: str = "en",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.max_new_lists = max_new_lists
        self.concurrent_calls = concurrent_calls
        self.language = language
        self._new_lists_created = 0

    # ------------------------------------------------------------------
    # Core: Learn-then-classify
    # ------------------------------------------------------------------

    def categorize_repos(
        self,
        uncategorized: list[Repository],
        existing_lists: list[StarList],
        second_pass_repos: Optional[list[Repository]] = None,
    ) -> list[Categorization]:
        """Main entry: categorize uncategorized repos by learning from existing lists.

        Args:
            uncategorized: Repos not in any list.
            existing_lists: All existing Star Lists with their repos (training data).
            second_pass_repos: Repos to re-categorize with README context.

        Returns:
            List of categorization decisions.
        """
        if not uncategorized:
            logger.info("No uncategorized repos to process.")
            return []

        is_cold_start = len(existing_lists) == 0
        system_prompt = self._build_system_prompt(
            existing_lists, len(uncategorized), is_cold_start
        )

        # Split into batches
        batches = [
            uncategorized[i : i + self.batch_size]
            for i in range(0, len(uncategorized), self.batch_size)
        ]

        logger.info(
            "Categorizing %d repos in %d batches (batch_size=%d, concurrent=%d)",
            len(uncategorized),
            len(batches),
            self.batch_size,
            self.concurrent_calls,
        )

        all_results: list[Categorization] = []

        # Parallel batch processing
        with ThreadPoolExecutor(max_workers=self.concurrent_calls) as executor:
            futures = {
                executor.submit(
                    self._process_batch, system_prompt, batch, i + 1, len(batches)
                ): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as exc:
                    logger.error("Batch %d failed: %s", batch_idx + 1, exc)

        # Second pass for low-confidence repos with README
        if second_pass_repos:
            logger.info(
                "Running second pass for %d low-confidence repos with README context",
                len(second_pass_repos),
            )
            second_pass_results = self._second_pass(system_prompt, second_pass_repos)
            # Replace low-confidence results with second-pass results
            second_pass_map = {c.repo_full_name: c for c in second_pass_results}
            all_results = [
                second_pass_map.get(c.repo_full_name, c) for c in all_results
            ]

        logger.info(
            "Categorization complete: %d results (%d high, %d medium, %d low confidence)",
            len(all_results),
            len([c for c in all_results if c.confidence == Confidence.HIGH]),
            len([c for c in all_results if c.confidence == Confidence.MEDIUM]),
            len([c for c in all_results if c.confidence == Confidence.LOW]),
        )

        return all_results

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        existing_lists: list[StarList],
        total_uncategorized: int,
        is_cold_start: bool,
    ) -> str:
        zh = self.language.startswith("zh")

        if is_cold_start:
            tpl = SYSTEM_PROMPT_COLD_START_ZH if zh else SYSTEM_PROMPT_COLD_START
            return tpl.format(
                total_stars=total_uncategorized,
                max_new_lists=self.max_new_lists,
            )

        lists_text = "\n\n".join(
            sl.summary_for_llm(max_repos=15) for sl in existing_lists
        )
        tpl = SYSTEM_PROMPT_LEARN_ZH if zh else SYSTEM_PROMPT_LEARN
        return tpl.format(
            existing_lists=lists_text,
            list_count=len(existing_lists),
            max_new_lists=self.max_new_lists,
        )

    def _build_batch_prompt(
        self, repos: list[Repository], include_readme: bool = False
    ) -> str:
        summaries = "\n".join(r.summary(include_readme=include_readme) for r in repos)
        tpl = (
            USER_PROMPT_BATCH_ZH
            if self.language.startswith("zh")
            else USER_PROMPT_BATCH
        )
        return tpl.format(count=len(repos), repo_summaries=summaries)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        system_prompt: str,
        batch: list[Repository],
        batch_num: int,
        total_batches: int,
    ) -> list[Categorization]:
        logger.info(
            "Processing batch %d/%d (%d repos)", batch_num, total_batches, len(batch)
        )

        user_prompt = self._build_batch_prompt(batch)
        raw = self._call_llm(system_prompt, user_prompt)
        results = self._parse_response(raw, batch)

        logger.info(
            "Batch %d/%d: %d categorizations parsed",
            batch_num,
            total_batches,
            len(results),
        )
        return results

    def _second_pass(
        self,
        system_prompt: str,
        repos: list[Repository],
    ) -> list[Categorization]:
        """Re-categorize repos using README context."""
        summaries = "\n".join(r.summary(include_readme=True) for r in repos)
        tpl = (
            USER_PROMPT_SECOND_PASS_ZH
            if self.language.startswith("zh")
            else USER_PROMPT_SECOND_PASS
        )
        user_prompt = tpl.format(repo_summaries=summaries)
        raw = self._call_llm(system_prompt, user_prompt)
        return self._parse_response(raw, repos)

    # ------------------------------------------------------------------
    # Smart features
    # ------------------------------------------------------------------

    def detect_stale_repos(self, repos: list[Repository]) -> list[dict[str, Any]]:
        candidates = [
            r
            for r in repos
            if r.is_archived or (r.days_since_pushed and r.days_since_pushed > 365)
        ]
        if not candidates:
            return []

        summaries = "\n".join(r.summary() for r in candidates[:50])
        zh = self.language.startswith("zh")
        tpl = PROMPT_STALE_DETECTION_ZH if zh else PROMPT_STALE_DETECTION
        prompt = tpl.format(repo_summaries=summaries)
        analyst = (
            "你是一位 GitHub 仓库分析师。"
            if zh
            else "You are a GitHub repository analyst."
        )
        raw = self._call_llm(analyst, prompt)

        try:
            result = self._extract_json(raw)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse stale detection response")
            return []

    def detect_duplicates(self, repos: list[Repository]) -> list[dict[str, Any]]:
        sample = repos[:80]
        summaries = "\n".join(r.summary() for r in sample)
        zh = self.language.startswith("zh")
        tpl = PROMPT_DUPLICATE_DETECTION_ZH if zh else PROMPT_DUPLICATE_DETECTION
        prompt = tpl.format(repo_summaries=summaries)
        analyst = (
            "你是一位 GitHub 仓库分析师。"
            if zh
            else "You are a GitHub repository analyst."
        )
        raw = self._call_llm(analyst, prompt)

        try:
            result = self._extract_json(raw)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse duplicate detection response")
            return []

    def check_recategorization(
        self,
        existing_lists: list[StarList],
    ) -> list[dict[str, Any]]:
        if not existing_lists:
            return []

        categorized_lines = []
        for sl in existing_lists:
            for repo in sl.repos[:10]:
                categorized_lines.append(f'- {repo.full_name} → "{sl.name}"')

        list_defs = "\n".join(
            f'- "{sl.name}": {sl.description or "(no description)"} ({sl.repo_count} repos)'
            for sl in existing_lists
        )

        zh = self.language.startswith("zh")
        tpl = PROMPT_RECATEGORIZE_CHECK_ZH if zh else PROMPT_RECATEGORIZE_CHECK
        prompt = tpl.format(
            categorized_summaries="\n".join(categorized_lines),
            list_definitions=list_defs,
        )
        reviewer = (
            "你是一位 GitHub 仓库分类审查员。"
            if zh
            else "You are a GitHub repository categorization reviewer."
        )
        raw = self._call_llm(reviewer, prompt)

        try:
            result = self._extract_json(raw)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse recategorization check response")
            return []

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return raw text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        source_repos: list[Repository],
    ) -> list[Categorization]:
        """Parse LLM JSON response into Categorization objects."""
        try:
            items = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "Failed to parse LLM response as JSON: %s\nRaw: %s", exc, raw[:500]
            )
            return []

        if not isinstance(items, list):
            logger.error("LLM response is not a JSON array: %s", type(items))
            return []

        # Build lookup for node IDs
        repo_id_map = {r.full_name: r.node_id for r in source_repos}

        results: list[Categorization] = []
        for item in items:
            repo_name = item.get("repo", "")
            node_id = repo_id_map.get(repo_name, "")

            if not node_id:
                logger.warning(
                    "Repo '%s' from LLM response not found in source repos", repo_name
                )
                continue

            confidence_str = item.get("confidence", "medium").lower()
            try:
                confidence = Confidence(confidence_str)
            except ValueError:
                confidence = Confidence.MEDIUM

            is_new = item.get("new_list", False)
            if is_new:
                self._new_lists_created += 1
                if self._new_lists_created > self.max_new_lists:
                    logger.warning(
                        "Max new lists (%d) reached — assigning '%s' to skip",
                        self.max_new_lists,
                        repo_name,
                    )
                    continue

            results.append(
                Categorization(
                    repo_full_name=repo_name,
                    repo_node_id=node_id,
                    list_name=item.get("list", "Uncategorized"),
                    reason=item.get("reason", ""),
                    confidence=confidence,
                    is_new_list=is_new,
                    new_list_description=item.get("new_list_description", ""),
                )
            )

        return results

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()

        # Strip <think>...</think> blocks (reasoning models like MiniMax-M2.1)
        think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
        if think_match:
            text = think_match.group(1).strip()

        # Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        return json.loads(text)
