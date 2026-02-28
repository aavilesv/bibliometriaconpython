import re
import scrapy
from urllib.parse import urljoin


OJS_ISSUE_ARCHIVE_SUFFIX = "/issue/archive"


class OJSPortalSpider(scrapy.Spider):
    name = "ojs_portals"

    start_urls = [
        "https://revistas.ufrj.br/",
        "https://revistas.face.ufmg.br/",
        "https://estudiosrurales.unq.edu.ar/index.php/ER",
    ]

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.5,
        "CONCURRENT_REQUESTS": 2,
        "AUTOTHROTTLE_ENABLED": True,
        "FEEDS": {
            "output.csv": {
                "format": "csv",
                "encoding": "utf8",
                "fields": [
                    "journal",
                    "year",
                    "title",
                    "authors",
                    "doi",
                    "abstract",
                    "pdf",
                    "url"
                ]
            },
        },
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "AcademicCrawler/1.0"
        }
    }

    # ========================
    # PORTAL
    # ========================

    def parse(self, response):
        journal_links = set()

        for href in response.css("a::attr(href)").getall():
            full = urljoin(response.url, href)

            if "/index.php/" in full:
                if not re.search(r"\.(jpg|png|css|js|pdf)$", full, re.I):
                    journal_links.add(full)

        if journal_links:
            for jurl in sorted(journal_links):
                yield scrapy.Request(jurl, callback=self.parse_journal)

    # ========================
    # JOURNAL
    # ========================

    def parse_journal(self, response):
        journal_name = response.css("title::text").get(default="").strip()

        archive_url = response.url.rstrip("/") + OJS_ISSUE_ARCHIVE_SUFFIX
        yield scrapy.Request(
            archive_url,
            callback=self.parse_issue_archive,
            meta={"journal": journal_name}
        )

    # ========================
    # ISSUE ARCHIVE
    # ========================

    def parse_issue_archive(self, response):
        journal_name = response.meta.get("journal")

        for href in response.css("a::attr(href)").getall():
            full = urljoin(response.url, href)

            if "/issue/view/" in full:
                yield scrapy.Request(
                    full,
                    callback=self.parse_issue,
                    meta={"journal": journal_name}
                )

    # ========================
    # ISSUE
    # ========================

    def parse_issue(self, response):
        journal_name = response.meta.get("journal")

        for href in response.css("a::attr(href)").getall():
            full = urljoin(response.url, href)

            if "/article/view/" in full:
                yield scrapy.Request(
                    full,
                    callback=self.parse_article,
                    meta={"journal": journal_name}
                )

    # ========================
    # ARTICLE
    # ========================

    def parse_article(self, response):
        journal_name = response.meta.get("journal")

        title = response.css("h1::text").get(default="").strip()

        authors = ", ".join(
            [a.strip() for a in response.css(".authors *::text").getall() if a.strip()]
        )

        abstract = " ".join(
            [t.strip() for t in response.css(".abstract *::text").getall() if t.strip()]
        )

        doi = response.css("a[href*='doi.org']::attr(href)").get()

        pdf = None
        for href in response.css("a::attr(href)").getall():
            full = urljoin(response.url, href)
            if re.search(r"/article/(download|view)/", full) and full.lower().endswith(".pdf"):
                pdf = full
                break

        year_match = re.search(r"(20\d{2})", response.text)
        year = year_match.group(1) if year_match else None

        yield {
            "journal": journal_name,
            "year": year,
            "title": title,
            "authors": authors,
            "doi": doi,
            "abstract": abstract,
            "pdf": pdf,
            "url": response.url,
        }
