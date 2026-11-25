"""Retry only broken connections, with fresh sessions and a fixed bound.

Read timeouts and HTTP/score failures propagate instead of duplicating long GPU
jobs. Callers retain shared audio after a reset because a handler may still run.
"""
import logging
import time
import requests

def post_with_retry(session_factory, reset_session, url, payload, timeout,
                    retries=2, backoff=0.5):
    for attempt in range(retries + 1):
        try:
            response = session_factory().post(url, json=payload, timeout=timeout)
            return response, attempt
        except requests.ConnectionError:
            reset_session()
            if attempt == retries:
                raise
            logging.warning('Reward connection reset; retry %d/%d with fresh session',
                            attempt + 1, retries)
            time.sleep(backoff * (attempt + 1))
