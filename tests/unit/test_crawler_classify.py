"""Error classification for the Playwright crawler (no browser)."""

from __future__ import annotations

from data_pipeline.crawler.crawler import classify_crawl_error


def test_classify_dns_from_message():
    msg = "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://dead.invalid/foo"
    cat, perm, human = classify_crawl_error(RuntimeError("x"), message=msg)
    assert cat == "dns"
    assert perm is True
    assert "DNS" in human


def test_classify_timeout_exceeded_string():
    msg = "Page.goto: Timeout 30000ms exceeded.\nCall log:"
    cat, perm, human = classify_crawl_error(RuntimeError("x"), message=msg)
    assert cat == "timeout"
    assert perm is False


def test_classify_connection_refused():
    msg = "net::ERR_CONNECTION_REFUSED at https://x/"
    cat, perm, _ = classify_crawl_error(RuntimeError("x"), message=msg)
    assert cat == "connection_refused"
    assert perm is False


def test_classify_ssl_permanent():
    msg = "net::ERR_CERT_AUTHORITY_INVALID at https://x/"
    cat, perm, _ = classify_crawl_error(RuntimeError("x"), message=msg)
    assert cat == "ssl"
    assert perm is True


def test_classify_generic_net_err():
    msg = "Page.goto: net::ERR_FAILED at https://x/"
    cat, perm, human = classify_crawl_error(RuntimeError("x"), message=msg)
    assert cat == "network"
    assert perm is False
    assert "ERR_FAILED" in human or "network" in human.lower()
