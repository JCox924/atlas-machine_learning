#!/usr/bin/env python3
"""
Module that queries the SWAPI API to find starships
    available for a given passenger count.
"""
import requests


def availableShips(passengerCount):
    """
    Return a list of starship names that can hold
        at least passengerCount passengers.

    Args:
        passengerCount (int): Number of
            passengers to accommodate.

    Returns:
        list: Names of starships (str) that can
            carry at least passengerCount passengers.
    """
    ships = []
    url = 'https://swapi.dev/api/starships/'

    while url:
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            break
        data = response.json()

        for ship in data.get('results', []):
            passengers = ship.get('passengers', '0').replace(',', '')
            try:
                capacity = int(passengers)
            except ValueError:
                continue
            if capacity >= passengerCount:
                ships.append(ship.get('name'))

        url = data.get('next')

    return ships
