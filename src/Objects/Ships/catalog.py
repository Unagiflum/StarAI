"""Shared ship and ability definitions loaded from the JSON catalogs."""

import json

import src.const as const


def _load_catalog(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


SHIPS_DATA = _load_catalog(const.SHIPS_JSON_PATH)
ABILITIES_DATA = _load_catalog(const.ABILITIES_JSON_PATH)
