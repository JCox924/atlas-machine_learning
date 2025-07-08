#!/usr/bin/env python3
"""30-all module."""

def list_all(mongo_collection):
    """List all documents in a collection.

    Args:
        mongo_collection (pymongo.collection.Collection): The collection to query.
    Returns:
        List of all documents in the collection (empty list if none).
    """
    return list(mongo_collection.find())
