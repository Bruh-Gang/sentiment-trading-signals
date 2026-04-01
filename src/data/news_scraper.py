"""
Financial News Scraper
=======================
Collects financial news headlines from RSS feeds and financial news APIs.
Uses requests + BeautifulSoup for RSS parsing.

Requirements:
    pip install requests beautifulsoup4 feedparser
"""

import time
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Represents a single news headline with metadata."""
    title: str
    ticker: str
    source: str
    published: datetime
    url: str
    summary: Optional[str] = None
    sentiment_score: Optional[float] = None


RSS_FEEDS = {
    "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "marketwatch":   "https://www.marketwatch.com/rss/topstories",
    "seeking_alpha": "https://seekingalpha.com/api/sa/combined/{ticker}.xml",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


class FinancialNewsScraper:
    """
    Scrapes financial news headlines from multiple RSS sources.

    Attributes:
        tickers:    Stock tickers to scrape news for.
        lookback:   Number of days to look back.
        rate_limit: Seconds between requests (politeness).
    """

    def __init__(
        self,
        tickers: List[str],
        lookback_days: int = 5,
        rate_limit: float = 1.0,
        timeout: int = 10,
    ) -> None:
        self.tickers     = [t.upper() for t in tickers]
        self.lookback    = lookback_days
        self.rate_limit  = rate_limit
        self.timeout     = timeout
        self._cutoff     = datetime.utcnow() - timedelta(days=lookback_days)
        self.session     = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_rss(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse an RSS feed URL.

        Args:
            url: RSS feed URL.

        Returns:
            Parsed BeautifulSoup object, or None on failure.
        """
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "xml")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def parse_yahoo_rss(self, ticker: str) -> List[NewsItem]:
        """
        Parse Yahoo Finance RSS feed for a given ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of NewsItem objects.
        """
        url  = RSS_FEEDS["yahoo_finance"].format(ticker=ticker)
        soup = self.fetch_rss(url)
        if not soup:
            return []

        items: List[NewsItem] = []
        for item in soup.find_all("item"):
            title_tag = item.find("title")
            link_tag  = item.find("link")
            pub_tag   = item.find("pubDate")
            desc_tag  = item.find("description")

            if not title_tag:
                continue

            title   = title_tag.get_text(strip=True)
            url_str = link_tag.get_text(strip=True) if link_tag else ""
            summary = self._strip_html(desc_tag.get_text()) if desc_tag else None

            # Parse publication date
            published = datetime.utcnow()
            if pub_tag:
                try:
                    from email.utils import parsedate_to_datetime
                    published = parsedate_to_datetime(pub_tag.get_text()).replace(tzinfo=None)
                except Exception:
                    pass

            if published < self._cutoff:
                continue

            items.append(NewsItem(
                title=title,
                ticker=ticker,
                source="Yahoo Finance",
                published=published,
                url=url_str,
                summary=summary,
            ))

        time.sleep(self.rate_limit)
        return items

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.compile(r'<[^>]+>')
        return clean.sub('', text).strip()

    def scrape_all(self) -> List[NewsItem]:
        """
        Scrape news for all configured tickers.

        Returns:
            Combined list of NewsItem objects, sorted by date (newest first).
        """
        all_news: List[NewsItem] = []
        for ticker in self.tickers:
            logger.info(f"Scraping news for {ticker}")
            items = self.parse_yahoo_rss(ticker)
            all_news.extend(items)
            logger.info(f"  Found {len(items)} items for {ticker}")

        all_news.sort(key=lambda x: x.published, reverse=True)
        return all_news

    def to_dataframe(self, news: List[NewsItem]):
        """Convert news items to a pandas DataFrame."""
        import pandas as pd
        records = [
            {
                "title":     n.title,
                "ticker":    n.ticker,
                "source":    n.source,
                "published": n.published,
                "url":       n.url,
                "summary":   n.summary,
            }
            for n in news
        ]
        return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = FinancialNewsScraper(tickers=["AAPL", "MSFT"], lookback_days=3)
    news = scraper.scrape_all()
    print(f"Total articles fetched: {len(news)}")
    for item in news[:5]:
        print(f"[{item.ticker}] {item.published.date()} — {item.title[:80]}")
