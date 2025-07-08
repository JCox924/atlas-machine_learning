#!/usr/bin/env python3
"""32-update_topics module."""


def update_topics(mongo_collection, name, topics):
    """Update the topics list for all school documents matching the given name.

    Args:
        mongo_collection (pymongo.collection.Collection): The collection to update.
        name (str): The school name to match.
        topics (list of str): The new list of topics.
    """
    mongo_collection.update_many(
        { "name": name },
        { "$set": { "topics": topics } }
    )
