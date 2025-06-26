#!/usr/bin/env python3
"""
Script that fetches and prints the location of a GitHub user from their API URL.
"""

import sys
import time
import requests


def main():
    """
    Main execution: fetch and print GitHub user location.

    Behavior:
        - If status code is 200: prints the 'location' field.
        - If status code is 404: prints "Not found".
        - If status code is 403: prints "Reset in X min" where X is minutes
          until rate limit reset (from 'X-RateLimit-Reset' header).
    """
    if len(sys.argv) < 2:
        sys.exit(1)

    url = sys.argv[1]
    response = requests.get(url, verify=False)
    status = response.status_code

    if status == 404:
        print("Not found")
        return

    if status == 403:
        reset = response.headers.get('X-RateLimit-Reset')
        try:
            reset = int(reset)
            now = int(time.time())
            minutes = int((reset - now) / 60) if reset > now else 0
        except Exception:
            minutes = 0
        print(f"Reset in {minutes} min")
        return

    if status == 200:
        data = response.json()
        print(data.get('location'))
        return

    print("Not found")


if __name__ == '__main__':
    main()
