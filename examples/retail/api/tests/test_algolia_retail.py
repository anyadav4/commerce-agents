# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Algolia-served search for the retail backend, against a fake Algolia endpoint."""

import json

import httpx
import pytest

from commerce_common.skills import SkillRegistry
from retail.api.algolia_retail import (
    INDEX_SETTINGS,
    AlgoliaConfig,
    AlgoliaRetail,
    index_records,
    search_params,
)
from shopping_agent import SearchFilters
from shopping_agent.executor import ShoppingToolExecutor

CONFIG = AlgoliaConfig(app_id="APP", search_key="search-only", index="acme")


def make_backend(handler) -> tuple[AlgoliaRetail, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return AlgoliaRetail(CONFIG, client=client), seen


def hits(*ids: str):
    return lambda request: httpx.Response(200, json={"hits": [{"objectID": i} for i in ids]})


async def test_hits_resolve_to_catalog_records_in_the_engines_order(session):
    backend, seen = make_backend(hits("AR-1202", "AR-1201"))
    found = await backend.search_products(session, "tent", limit=5)
    assert [p.product_id for p in found] == ["AR-1202", "AR-1201"]
    assert found[0].price == backend.products["AR-1202"].price
    request = seen[0]
    assert str(request.url) == "https://app-dsn.algolia.net/1/indexes/acme/query"
    assert request.headers["X-Algolia-API-Key"] == "search-only"
    assert json.loads(request.content)["hitsPerPage"] == 5


async def test_an_id_the_catalog_lacks_is_dropped(session):
    backend, _ = make_backend(hits("NOT-IN-CATALOG", "AR-1201"))
    found = await backend.search_products(session, "tent")
    assert [p.product_id for p in found] == ["AR-1201"]


async def test_a_price_moved_since_indexing_is_filtered_on_the_current_price(session):
    backend, _ = make_backend(hits("AR-1201"))
    backend.products["AR-1201"].price = 500.0
    found = await backend.search_products(session, "tent", SearchFilters(max_price=300))
    assert found == []


async def test_no_hits_is_an_empty_result_not_a_fallback(session):
    backend, seen = make_backend(hits())
    assert await backend.search_products(session, "zzzqqq") == []
    assert len(seen) == 1


@pytest.mark.parametrize(
    "failure",
    [
        lambda request: httpx.Response(503),
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json={"message": "no hits key"}),
        lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout("slow")),
    ],
)
async def test_a_failing_engine_falls_back_to_the_catalog_search(session, failure):
    backend, _ = make_backend(failure)
    found = await backend.search_products(session, "tent for two people")
    assert found and found[0].product_id in {"AR-1201", "AR-1202"}


async def test_a_blank_query_does_not_call_the_engine(session):
    backend, seen = make_backend(hits("AR-1201"))
    assert await backend.search_products(session, "  ") == []
    assert seen == []


async def test_sorts_other_than_relevance_reorder_a_larger_pool(session):
    backend, seen = make_backend(hits("AR-1201", "AR-1202", "AR-1207"))
    found = await backend.search_products(session, "tent", SearchFilters(sort="price_asc"), limit=2)
    assert [p.price for p in found] == sorted(p.price for p in found)
    assert len(found) == 2
    assert json.loads(seen[0].content)["hitsPerPage"] == 50


def test_filters_become_numeric_and_optional_filters():
    body = search_params(
        "tent",
        SearchFilters(
            min_price=20,
            max_price=250,
            min_rating=4,
            category="Outdoor",
            attributes={"color": 'a"b'},
        ),
        8,
    )
    assert body["numericFilters"] == ["price>=20.0", "price<=250.0", "rating>=4.0"]
    assert body["optionalFilters"] == ['category:"outdoor"', 'attribute_pairs:"color=ab"']
    assert "numericFilters" not in search_params("tent", SearchFilters(), 8)


def test_config_needs_all_three_variables(monkeypatch):
    for name in ("ALGOLIA_APP_ID", "ALGOLIA_SEARCH_KEY", "ALGOLIA_INDEX"):
        monkeypatch.setenv(name, "x")
    assert AlgoliaConfig.from_env() == AlgoliaConfig("x", "x", "x")
    monkeypatch.delenv("ALGOLIA_INDEX")
    assert AlgoliaConfig.from_env() is None
    assert AlgoliaRetail.from_env() is None


def test_index_records_cover_every_listing_and_none_of_the_variants(backend):
    records = index_records(backend.products.values())
    ids = {record["objectID"] for record in records}
    assert ids == set(backend.products) and not ids & set(backend.variants)
    mattress = next(r for r in records if r["objectID"] == "AR-1902")
    assert "king" in mattress["options_text"]
    assert all(set(INDEX_SETTINGS["attributesForFaceting"]) - {"brand"} <= set(r) for r in records)


async def test_an_engine_result_flows_through_the_agents_cart_gates(main, session, state):
    backend, _ = make_backend(hits("AR-1201", "AR-1902"))
    executor = ShoppingToolExecutor(
        backend=backend,
        config=main.agent.config,
        skills=SkillRegistry([]),
        session=session,
        state=state,
        extensions=list(main.agent.extra_presentation_tools),
    )
    unseen = await executor.execute("add_to_cart", {"product_id": "AR-1201", "quantity": 1})
    assert unseen.blocked == "provenance"
    await executor.execute("search_products", {"query": "tent"})
    added = await executor.execute("add_to_cart", {"product_id": "AR-1201", "quantity": 1})
    assert not added.refused
    assert [item.product_id for item in (await backend.get_cart(session)).items] == ["AR-1201"]
    family = await executor.execute("add_to_cart", {"product_id": "AR-1902", "quantity": 1})
    assert family.blocked == "options"


async def test_a_search_logs_counts_at_info_and_the_query_only_at_debug(session, caplog):
    backend, _ = make_backend(hits("AR-1201", "NOT-IN-CATALOG"))
    with caplog.at_level("DEBUG", logger="retail.api.algolia_retail"):
        await backend.search_products(session, "secret words")
    info = next(r for r in caplog.records if r.getMessage().startswith("algolia search "))
    assert info.levelname == "INFO"
    assert "hits=2 kept=1" in info.getMessage() and "status=200" in info.getMessage()
    assert session.session_id not in caplog.text and "search-only" not in caplog.text
    assert "secret words" not in info.getMessage()
    assert any("secret words" in r.getMessage() and r.levelname == "DEBUG" for r in caplog.records)
