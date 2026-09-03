"""Pull granule files into a local cache, in parallel and only once."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

log = logging.getLogger(__name__)

CHUNK = 1 << 18


def download(urls, cache_dir, max_workers=10, retries=3, timeout=60, session=None):
    """Download each URL into cache_dir, skipping what is already there.

    Returns local paths in the order given. Credentials come from an Earthdata login in
    ~/.netrc, which the ASF data pool requires: without one every request answers 403.

    Most of the cost is the TCP and TLS handshake rather than the transfer, so one
    keep-alive session is shared across the pool.
    """
    urls = list(urls)
    if retries < 1:
        raise ValueError(f"retries is how many attempts to make, so at least 1, not {retries}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # One pooled, authenticated session: the handshake costs more than the transfer.
    if session is None:
        import asf_search as asf
        import requests

        session = asf.ASFSession()
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=max_workers, pool_maxsize=max_workers, max_retries=0))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fetched = pool.map(lambda url: _fetch(session, url, cache_dir, retries, timeout), urls)
        paths = list(tqdm(fetched, total=len(urls), desc="downloading"))

    log.info("%d files in %s", len(paths), cache_dir)
    return paths


def _fetch(session, url, cache_dir, retries, timeout):
    """One file into the cache, retried, returning where it landed."""
    path = cache_dir / Path(url).name
    if path.exists() and path.stat().st_size:
        return path

    # An interrupted transfer must not land where the check above reads it as cached.
    part = path.with_name(path.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            _stream_to(session, url, part, timeout)
            return part.replace(path)
        except Exception as err:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            log.warning("retry %d of %d for %s: %s", attempt, retries, path.name, err)


def _stream_to(session, url, part, timeout):
    """Write the response body out, and check that all of it arrived."""
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        # requests decodes gzip as it streams, so the declared length matches the bytes on
        # disk only when the body arrives as it was sent.
        encoded = "Content-Encoding" in response.headers
        expected = None if encoded else response.headers.get("Content-Length")

        with open(part, "wb") as out:
            for chunk in response.iter_content(CHUNK):
                out.write(chunk)

    written = part.stat().st_size
    if expected is not None and written != int(expected):
        raise OSError(f"{part.name}: got {written} of {expected} bytes")
