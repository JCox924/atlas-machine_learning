#!/usr/bin/env python3
"""31-insert_school module."""

def insert_school(mongo_collection, **kwargs):
    """Insert a new document in a collection based on kwargs.

    Args:
        mongo_collection (pymongo.collection.Collection): The collection to insert into.
        **kwargs: Field names and values for the document.
    Returns:
        The inserted document's _id.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
