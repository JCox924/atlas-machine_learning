#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch
    information using the unofficial SpaceX API.
"""
import requests


def get_first_launch():
    """
    Retrieve and return the earliest launch
        (by date_unix) from the SpaceX API.

    Returns:
        dict: Launch data.
    """
    launches_url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(launches_url, verify=False)
    response.raise_for_status()
    launches = response.json()
    # Sort by date_unix ascending
    launches.sort(key=lambda x: x.get('date_unix', 0))
    return launches[0]


def main():
    # Get the first launch
    launch = get_first_launch()
    name = launch.get('name')
    date_local = launch.get('date_local')

    # Fetch rocket name
    rocket_id = launch.get('rocket')
    rocket_resp = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}', verify=False)
    rocket_resp.raise_for_status()
    rocket_name = rocket_resp.json().get('name')

    # Fetch launchpad info
    pad_id = launch.get('launchpad')
    pad_resp = requests.get(f'https://api.spacexdata.com/v4/launchpads/{pad_id}', verify=False)
    pad_resp.raise_for_status()
    pad_data = pad_resp.json()
    pad_name = pad_data.get('name')
    pad_locality = pad_data.get('locality')

    # Print in the required format
    print(f"{name} ({date_local}) {rocket_name} - {pad_name} ({pad_locality})")


if __name__ == '__main__':
    main()
