#!/usr/bin/env python3
"""
Script that prints the location of a GitHub user via the GitHub API.
"""

import sys
import time
import requests


def fetch_location(url):
    """
    Fetch the user's location from the GitHub API URL.

    Returns:
        tuple(status_code, location_or_reset)
    """
    response = requests.get(url)
    status = response.status_code

    if status == 200:
        data = response.json()
        return 200, data.get('location') if data.get('location') is not None else ''
    if status == 404:
        return 404, None
    if status == 403:
        reset = response.headers.get('X-RateLimit-Reset')
        try:
            reset = int(reset)
            now = int(time.time())
            minutes = (reset - now) // 60 if reset > now else 0
        except Exception:
            minutes = 0
        return 403, minutes
    return status, None


def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    url = sys.argv[1]
    status, info = fetch_location(url)

    if status == 200:
        print(info)
    elif status == 404:
        print("Not found")
    elif status == 403:
        print(f"Reset in {info} min")
    else:
        print("Not found")


if __name__ == '__main__':
    main()
