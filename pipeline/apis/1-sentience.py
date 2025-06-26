#!/usr/bin/env python3
"""
Module to retrieve home planets of all sentient species from the Star Wars API (SWAPI).
"""
import requests


def sentientPlanets():
    """
    Return a list of unique planet names that are homeworlds of sentient species.

    A species is considered sentient if its classification or designation is 'sentient'.
    Results are ordered by ascending planet ID.

    Returns:
        list: Planet names (str).
    """
    species_url = 'https://swapi.dev/api/species/'
    homeworld_urls = []

    # Traverse all species pages
    while species_url:
        response = requests.get(species_url, verify=False)
        if response.status_code != 200:
            break
        data = response.json()

        for species in data.get('results', []):
            classification = species.get('classification', '').strip().lower()
            designation = species.get('designation', '').strip().lower()
            hw = species.get('homeworld')
            if (classification == 'sentient' or designation == 'sentient') and hw:
                homeworld_urls.append(hw)

        species_url = data.get('next')

    unique_urls = set(homeworld_urls)

    def planet_id(url):
        try:
            return int(url.rstrip('/').split('/')[-1])
        except Exception:
            return float('inf')

    sorted_urls = sorted(unique_urls, key=planet_id)

    planets = []
    for url in sorted_urls:
        resp = requests.get(url, verify=False)
        if resp.status_code != 200:
            continue
        planet_data = resp.json()
        name = planet_data.get('name')
        if name:
            planets.append(name)

    return planets
