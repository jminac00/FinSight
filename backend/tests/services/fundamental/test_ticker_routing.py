"""Tests for the ADR/local-listing routing recommendations."""

from app.services.fundamental.engine.ticker_routing import (
    recommend_ticker,
    registered_route_tickers,
)


def test_adr_recommends_local_listing():
    rec = recommend_ticker("NVO")
    assert rec["preferred_ticker"] == "NOVO-B.CO"
    assert "NOVO-B.CO" in rec["alternative_tickers"]
    assert rec["whether_rerun_is_recommended"] is True


def test_asml_routes_to_amsterdam_listing():
    rec = recommend_ticker("ASML")
    assert rec["preferred_ticker"] == "ASML.AS"
    assert rec["whether_rerun_is_recommended"] is True


def test_unregistered_ticker_has_no_alternative():
    rec = recommend_ticker("AAPL")
    assert rec["preferred_ticker"] == "AAPL"
    assert rec["alternative_tickers"] == []
    assert rec["whether_rerun_is_recommended"] is False


def test_input_is_normalized_to_uppercase():
    rec = recommend_ticker("nvo")
    assert rec["input_ticker"] == "NVO"
    assert rec["preferred_ticker"] == "NOVO-B.CO"


def test_registered_groups_are_exposed():
    groups = registered_route_tickers()
    assert "NOVO-B.CO" in groups["NOVO"]
    assert "7203.T" in groups["TOYOTA"]
