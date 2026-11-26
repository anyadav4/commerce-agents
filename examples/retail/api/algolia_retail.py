# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""``AlgoliaRetail``: the retail backend with search served by an Algolia index. Algolia
only ranks: each hit's ``objectID`` is a catalog ``product_id``, and the record returned
is the in-memory one, so price, stock, variants, the cart, and fulfilment stay as
``MockRetail`` has them. ``index_records`` and ``INDEX_SETTINGS`` describe what
``scripts/load_algolia.py`` puts in the index.

Search calls Algolia's REST API over ``httpx``; a failed call falls back to the mock's
keyword search, so a search outage does not fail the turn.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from commerce_common.turn import elapsed_ms, session_tag
from demo_common.storefront_fixtures import (
    option_text,
    summary_of,
    within_price_and_rating,
)
from demo_common.tracing import trace_backend_call
from shopping_agent import Product, ProductDetails, SearchFilters, ShoppingSessionContext

from .mock_retail import DATA_DIR, MockRetail

logger = logging.getLogger(__name__)

# A sort other than relevance reorders a pool of the best matches, so "cheapest" is the
# cheapest of the pool rather than of the one page asked for.
SORT_POOL = 50

INDEX_SETTINGS: dict[str, Any] = {
    # Earlier entries rank higher; a comma joins attributes of equal rank.
    "searchableAttributes": [
        "title",
        "brand",
        "category",
        "tags",
        "attribute_text,options_text,color",
        "description",
    ],
    "attributesForFaceting": ["category", "brand", "attribute_pairs"],
    "customRanking": ["desc(rating)", "desc(review_count)"],
    "removeWordsIfNoResults": "lastWords",
}


@dataclass(frozen=True)
class AlgoliaConfig:
    app_id: str
    search_key: str
    index: str
    timeout: float = 3.0

    @classmethod
    def from_env(cls) -> AlgoliaConfig | None:
        """The config when ``ALGOLIA_APP_ID``, ``ALGOLIA_SEARCH_KEY`` and ``ALGOLIA_INDEX``
        are all set, else None."""
        app_id = os.environ.get("ALGOLIA_APP_ID", "")
        key = os.environ.get("ALGOLIA_SEARCH_KEY", "")
        index = os.environ.get("ALGOLIA_INDEX", "")
        return cls(app_id, key, index) if app_id and key and index else None

    @property
    def query_url(self) -> str:
        return f"https://{self.app_id}-dsn.algolia.net/1/indexes/{self.index}/query"

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Algolia-Application-Id": self.app_id, "X-Algolia-API-Key": self.search_key}


def _colors(product: ProductDetails) -> list[str]:
    """The colors a listing comes in: its own ``color`` attribute, or its color option."""
    values = [product.attributes["color"]] if "color" in product.attributes else []
    values += product.options.get("color", [])
    return list(dict.fromkeys(value.lower() for value in values))


def index_records(listings: Iterable[ProductDetails]) -> list[dict[str, Any]]:
    """One index record per listing (plain product or family; a family's variants are
    searchable through its option words, not as records of their own). The shape follows
    Algolia's sample commerce data (``hierarchical_categories``, ``product_type``,
    ``color``, ``tags``, ``showcase_image``) plus the fields search ranks and filters on."""
    records = []
    for product in listings:
        category = product.category or ""
        record: dict[str, Any] = {
            "objectID": product.product_id,
            "title": product.title,
            "brand": product.brand or "",
            "description": " ".join(
                text for text in (product.short_description, product.long_description) if text
            ),
            "category": category,
            "hierarchical_categories": {
                "lvl0": category,
                "lvl1": f"{category} > {product.brand}" if product.brand else category,
            },
            "product_type": category,
            "color": _colors(product),
            "tags": product.labels,
            "attribute_text": " ".join(f"{k} {v}" for k, v in product.attributes.items()),
            "attribute_pairs": [f"{k}={v}" for k, v in product.attributes.items()],
            "options_text": option_text(product),
            "price": product.price,
            "currency": product.currency,
            "rating": product.rating or 0,
            "review_count": product.review_count or 0,
            "in_stock": product.in_stock,
        }
        if product.image_url:
            record["showcase_image"] = product.image_url
        records.append(record)
    return records


def search_params(query: str, filters: SearchFilters, hits: int) -> dict[str, Any]:
    """The query body for one search. Price and rating are hard filters; category and
    attributes only boost, because the model guesses the taxonomy and a wrong guess
    should reorder results, not empty them."""
    numeric = []
    if filters.min_price is not None:
        numeric.append(f"price>={filters.min_price}")
    if filters.max_price is not None:
        numeric.append(f"price<={filters.max_price}")
    if filters.min_rating is not None:
        numeric.append(f"rating>={filters.min_rating}")
    optional = []
    if filters.category:
        optional.append(f'category:"{_quoted(filters.category.lower())}"')
    optional += [
        f'attribute_pairs:"{_quoted(f"{key}={value}")}"'
        for key, value in filters.attributes.items()
    ]
    body: dict[str, Any] = {
        "query": query,
        "hitsPerPage": hits,
        "attributesToRetrieve": ["objectID"],
        "attributesToHighlight": [],
        "attributesToSnippet": [],
    }
    if numeric:
        body["numericFilters"] = numeric
    if optional:
        body["optionalFilters"] = optional
    return body


def _trace_inputs(
    query: str, filters: SearchFilters | None = None, limit: int = 8, **_: Any
) -> dict[str, Any]:
    """What a trace records of a search: the words and filters, never the session."""
    return {
        "query": query,
        "filters": filters.model_dump(exclude_defaults=True) if filters else {},
        "limit": limit,
    }


def _quoted(value: str) -> str:
    return value.replace('"', "").replace("\\", "")


class AlgoliaRetail(MockRetail):
    def __init__(
        self,
        config: AlgoliaConfig,
        *,
        client: httpx.AsyncClient | None = None,
        data_dir: Path = DATA_DIR,
    ) -> None:
        super().__init__(data_dir)
        self._algolia = config
        self._http = client or httpx.AsyncClient(timeout=config.timeout)

    @classmethod
    def from_env(cls) -> AlgoliaRetail | None:
        config = AlgoliaConfig.from_env()
        return cls(config) if config else None

    @trace_backend_call("algolia_search", inputs=_trace_inputs)
    async def search_products(
        self,
        session: ShoppingSessionContext,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 8,
    ) -> list[Product]:
        if not query.strip():
            return []
        wanted = filters or SearchFilters()
        by_relevance = wanted.sort == "relevance"
        params = search_params(query, wanted, limit if by_relevance else SORT_POOL)
        tag = session_tag(session.session_id)
        started = time.monotonic()
        try:
            response = await self._http.post(
                self._algolia.query_url, json=params, headers=self._algolia.headers
            )
            response.raise_for_status()
            hits = response.json()["hits"]
        except (httpx.HTTPError, KeyError, ValueError):
            logger.warning(
                "algolia search failed, using the catalog search session=%s index=%s elapsed_ms=%d",
                tag,
                self._algolia.index,
                elapsed_ms(started),
                exc_info=True,
            )
            return await super().search_products(session, query, filters, limit)

        ranked: list[ProductDetails] = []
        for hit in hits:
            record = self.products.get(str(hit.get("objectID", "")))
            if record is None:
                logger.warning("Algolia returned %r, which is not in the catalog", hit)
            # The index holds the prices it was loaded with; the catalog holds today's.
            elif within_price_and_rating(record, wanted):
                ranked.append(record)
        # Counts at INFO; the customer's words and the filters only at DEBUG.
        logger.info(
            "algolia search session=%s index=%s status=%d hits=%d kept=%d elapsed_ms=%d",
            tag,
            self._algolia.index,
            response.status_code,
            len(hits),
            len(ranked),
            elapsed_ms(started),
        )
        logger.debug("algolia query session=%s params=%s", tag, json.dumps(params))
        if wanted.sort == "price_asc":
            ranked.sort(key=lambda product: product.price)
        elif wanted.sort == "price_desc":
            ranked.sort(key=lambda product: -product.price)
        elif wanted.sort == "rating":
            ranked.sort(key=lambda product: -(product.rating or 0))
        return [summary_of(product) for product in ranked[:limit]]
