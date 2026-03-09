"""GitHub GraphQL API client for Star Lists management."""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from .models import Repository, StarList

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GraphQL Queries
# ---------------------------------------------------------------------------

QUERY_VIEWER_LOGIN = """
query {
  viewer { login }
  rateLimit { remaining resetAt }
}
"""

QUERY_USER_LISTS = """
query FetchUserLists($username: String!, $cursor: String) {
  user(login: $username) {
    lists(first: 20, after: $cursor) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        name
        description
        slug
        items(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            ... on Repository {
              id
              name
              nameWithOwner
              description
              url
              stargazerCount
              primaryLanguage { name }
              repositoryTopics(first: 10) {
                nodes { topic { name } }
              }
              isArchived
              pushedAt
            }
          }
        }
      }
    }
  }
}
"""

QUERY_LIST_ITEMS_PAGE = """
query FetchListItems($listId: ID!, $cursor: String) {
  node(id: $listId) {
    ... on UserList {
      items(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          ... on Repository {
            id
            name
            nameWithOwner
            description
            url
            stargazerCount
            primaryLanguage { name }
            repositoryTopics(first: 10) {
              nodes { topic { name } }
            }
            isArchived
            pushedAt
          }
        }
      }
    }
  }
}
"""

QUERY_STARRED_REPOS = """
query FetchStarredRepos($cursor: String) {
  viewer {
    starredRepositories(first: 100, after: $cursor, orderBy: {field: STARRED_AT, direction: DESC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      edges {
        starredAt
        node {
          id
          name
          nameWithOwner
          description
          url
          stargazerCount
          forkCount
          isArchived
          primaryLanguage { name }
          repositoryTopics(first: 10) {
            nodes { topic { name } }
          }
          pushedAt
          updatedAt
          owner { login }
          licenseInfo { spdxId }
        }
      }
    }
  }
}
"""

QUERY_REPO_README = """
query FetchReadme($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    object(expression: "HEAD:README.md") {
      ... on Blob { text }
    }
  }
}
"""

MUTATION_CREATE_LIST = """
mutation CreateUserList($name: String!, $description: String, $isPrivate: Boolean!) {
  createUserList(input: { name: $name, description: $description, isPrivate: $isPrivate }) {
    list { id name slug description }
  }
}
"""

MUTATION_UPDATE_LISTS_FOR_ITEM = """
mutation UpdateUserListsForItem($itemId: ID!, $listIds: [ID!]!) {
  updateUserListsForItem(input: { itemId: $itemId, listIds: $listIds }) {
    item {
      ... on Repository { nameWithOwner }
    }
  }
}
"""

MUTATION_DELETE_LIST = """
mutation DeleteUserList($listId: ID!) {
  deleteUserList(input: { listId: $listId }) {
    user { login }
  }
}
"""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GitHubGraphQLClient:
    """Client for GitHub's GraphQL API with rate-limit awareness."""

    GRAPHQL_URL = "https://api.github.com/graphql"

    def __init__(self, token: str, rate_limit_buffer: int = 500):
        self.token = token
        self.rate_limit_buffer = rate_limit_buffer
        self._rate_remaining: Optional[int] = None
        self._rate_reset_at: Optional[str] = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"bearer {token}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Low-level GraphQL
    # ------------------------------------------------------------------

    def _execute(
        self,
        query: str,
        variables: Optional[dict] = None,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
    ) -> dict:
        """Execute a GraphQL query with retry and rate-limit handling."""
        payload: dict[str, object] = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(1, retry_attempts + 1):
            # Rate-limit guard
            if (
                self._rate_remaining is not None
                and self._rate_remaining < self.rate_limit_buffer
            ):
                logger.warning(
                    "Rate limit low (%d remaining). Waiting until reset...",
                    self._rate_remaining,
                )
                self._wait_for_reset()

            try:
                resp = self.session.post(self.GRAPHQL_URL, json=payload, timeout=30)
            except requests.RequestException as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s", attempt, retry_attempts, exc
                )
                if attempt < retry_attempts:
                    time.sleep(retry_delay * attempt)
                    continue
                raise

            # Update rate-limit bookkeeping from response headers
            if "X-RateLimit-Remaining" in resp.headers:
                self._rate_remaining = int(resp.headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in resp.headers:
                self._rate_reset_at = resp.headers["X-RateLimit-Reset"]

            if resp.status_code == 502:
                logger.warning(
                    "GitHub returned 502 (attempt %d/%d)", attempt, retry_attempts
                )
                if attempt < retry_attempts:
                    time.sleep(retry_delay * attempt)
                    continue

            resp.raise_for_status()
            body = resp.json()

            if "errors" in body:
                error_messages = [e.get("message", str(e)) for e in body["errors"]]
                error_str = "; ".join(error_messages)

                # Retryable errors
                if any("rate limit" in m.lower() for m in error_messages):
                    logger.warning("Rate limited. Waiting 60s...")
                    time.sleep(60)
                    if attempt < retry_attempts:
                        continue

                # Non-retryable → raise
                raise RuntimeError(f"GraphQL errors: {error_str}")

            return body.get("data", {})

        raise RuntimeError("All retry attempts exhausted")

    def _wait_for_reset(self) -> None:
        """Sleep until the rate-limit window resets."""
        if self._rate_reset_at:
            try:
                reset_ts = int(self._rate_reset_at)
                wait = max(0, reset_ts - int(time.time())) + 5
            except ValueError:
                wait = 60
        else:
            wait = 60
        logger.info("Sleeping %d seconds for rate-limit reset...", wait)
        time.sleep(wait)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_viewer_login(self) -> str:
        """Get the authenticated user's login name."""
        data = self._execute(QUERY_VIEWER_LOGIN)
        login = data["viewer"]["login"]
        if "rateLimit" in data:
            self._rate_remaining = data["rateLimit"]["remaining"]
        logger.info(
            "Authenticated as: %s (rate remaining: %s)", login, self._rate_remaining
        )
        return login

    def get_user_lists(self, username: str) -> list[StarList]:
        """Fetch all Star Lists and their repos for a user."""
        all_lists: list[StarList] = []
        cursor: Optional[str] = None

        while True:
            data = self._execute(
                QUERY_USER_LISTS, {"username": username, "cursor": cursor}
            )
            lists_data = data.get("user", {}).get("lists", {})

            for node in lists_data.get("nodes", []):
                star_list = self._parse_star_list(node)
                # If the list has more items, paginate
                items_info = node.get("items", {})
                if items_info.get("pageInfo", {}).get("hasNextPage"):
                    extra_repos = self._fetch_remaining_list_items(
                        node["id"],
                        items_info["pageInfo"]["endCursor"],
                    )
                    star_list.repos.extend(extra_repos)
                all_lists.append(star_list)

            page_info = lists_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info["endCursor"]

        logger.info("Fetched %d Star Lists", len(all_lists))
        return all_lists

    def _fetch_remaining_list_items(
        self, list_id: str, cursor: Optional[str]
    ) -> list[Repository]:
        """Paginate through remaining items in a Star List."""
        repos: list[Repository] = []
        while cursor:
            data = self._execute(
                QUERY_LIST_ITEMS_PAGE, {"listId": list_id, "cursor": cursor}
            )
            items = data.get("node", {}).get("items", {})
            for repo_node in items.get("nodes", []):
                if repo_node:
                    repos.append(self._parse_repo_node(repo_node))
            pi = items.get("pageInfo", {})
            cursor = (
                str(pi["endCursor"])
                if pi.get("hasNextPage") and pi.get("endCursor")
                else None
            )
        return repos

    def get_starred_repos(self, max_repos: int = 0) -> list[Repository]:
        """Fetch all starred repos for the authenticated user.

        Args:
            max_repos: Limit number of repos fetched. 0 = unlimited.
        """
        all_repos: list[Repository] = []
        cursor: Optional[str] = None

        while True:
            data = self._execute(QUERY_STARRED_REPOS, {"cursor": cursor})
            starred = data.get("viewer", {}).get("starredRepositories", {})
            total = starred.get("totalCount", 0)

            for edge in starred.get("edges", []):
                node = edge.get("node", {})
                if not node:
                    continue
                repo = self._parse_repo_node(node)
                repo.starred_at = edge.get("starredAt")
                all_repos.append(repo)

                if max_repos and len(all_repos) >= max_repos:
                    logger.info(
                        "Reached max_repos limit (%d/%d total)", max_repos, total
                    )
                    return all_repos

            page_info = starred.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info["endCursor"]

            logger.info("Fetched %d / %d starred repos...", len(all_repos), total)

        logger.info("Fetched %d starred repos total", len(all_repos))
        return all_repos

    def fetch_readme(
        self, owner: str, name: str, max_length: int = 500
    ) -> Optional[str]:
        """Fetch a truncated README for a single repo."""
        try:
            data = self._execute(QUERY_REPO_README, {"owner": owner, "name": name})
            blob = data.get("repository", {}).get("object")
            if blob and blob.get("text"):
                text = blob["text"]
                return text[:max_length] if len(text) > max_length else text
        except Exception as exc:
            logger.debug("Failed to fetch README for %s/%s: %s", owner, name, exc)
        return None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create_list(
        self, name: str, description: str = "", private: bool = False
    ) -> StarList:
        """Create a new Star List."""
        data = self._execute(
            MUTATION_CREATE_LIST,
            {"name": name, "description": description, "isPrivate": private},
        )
        list_data = data.get("createUserList", {}).get("list", {})
        logger.info("Created Star List: %s (id=%s)", name, list_data.get("id"))
        return StarList(
            node_id=list_data["id"],
            name=list_data.get("name", name),
            description=list_data.get("description", description),
            slug=list_data.get("slug", ""),
        )

    def update_repo_lists(self, repo_node_id: str, list_ids: list[str]) -> None:
        """Set the list memberships for a repo.

        IMPORTANT: This REPLACES all memberships. Always pass the COMPLETE
        set of desired list IDs (existing + new).
        """
        self._execute(
            MUTATION_UPDATE_LISTS_FOR_ITEM,
            {"itemId": repo_node_id, "listIds": list_ids},
        )

    def delete_list(self, list_id: str) -> None:
        """Delete a Star List."""
        self._execute(MUTATION_DELETE_LIST, {"listId": list_id})
        logger.info("Deleted Star List: %s", list_id)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_repo_node(node: dict) -> Repository:
        """Parse a GraphQL repo node into a Repository model."""
        topics = []
        for t in node.get("repositoryTopics", {}).get("nodes", []):
            topic_name = t.get("topic", {}).get("name")
            if topic_name:
                topics.append(topic_name)

        lang = node.get("primaryLanguage")
        language = lang.get("name", "") if lang else ""

        license_info = node.get("licenseInfo")
        license_id = license_info.get("spdxId") if license_info else None

        owner_data = node.get("owner", {})

        return Repository(
            node_id=node.get("id", ""),
            name=node.get("name", ""),
            full_name=node.get("nameWithOwner", ""),
            description=node.get("description") or "",
            url=node.get("url", ""),
            language=language,
            topics=topics,
            stargazer_count=node.get("stargazerCount", 0),
            fork_count=node.get("forkCount", 0),
            is_archived=node.get("isArchived", False),
            pushed_at=node.get("pushedAt"),
            updated_at=node.get("updatedAt"),
            license_id=license_id,
            owner=owner_data.get("login", ""),
        )

    @staticmethod
    def _parse_star_list(node: dict) -> StarList:
        """Parse a GraphQL list node into a StarList model."""
        repos = []
        for repo_node in node.get("items", {}).get("nodes", []):
            if repo_node:
                repos.append(GitHubGraphQLClient._parse_repo_node(repo_node))

        return StarList(
            node_id=node["id"],
            name=node.get("name", ""),
            description=node.get("description") or "",
            slug=node.get("slug", ""),
            repos=repos,
        )
