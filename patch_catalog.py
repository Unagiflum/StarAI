import re
with open('src/Objects/Ships/catalog.py', 'r', encoding='utf-8') as f:
    content = f.read()

field_insert = '''    silhouette_count: int | None = None
    silhouette_colors: tuple[tuple[int, int, int, int], ...] | None = None
    silhouette_distances: tuple[float, ...] | None = None
    anim_length: int | None = None
    circle_thickness: int | None = None
    circle_color: tuple[tuple[int, int, int, int], ...] | None = None'''
content = content.replace('    silhouette_count: int | None = None\n    silhouette_colors: tuple[tuple[int, int, int, int], ...] | None = None\n    silhouette_distances: tuple[float, ...] | None = None', field_insert)

json_key_insert = '''        "SILHOUETTE_COUNT": "silhouette_count",
        "SILHOUETTE_COLORS": "silhouette_colors",
        "SILHOUETTE_DIST": "silhouette_distances",
        "ANIM_LENGTH": "anim_length",
        "CIRCLE_THICKNESS": "circle_thickness",
        "CIRCLE_COLOR": "circle_color",'''
content = content.replace('        "SILHOUETTE_COUNT": "silhouette_count",\n        "SILHOUETTE_COLORS": "silhouette_colors",\n        "SILHOUETTE_DIST": "silhouette_distances",', json_key_insert)

optional_insert = '''        "SILHOUETTE_COUNT",
        "SILHOUETTE_COLORS",
        "SILHOUETTE_DIST",
        "ANIM_LENGTH",
        "CIRCLE_THICKNESS",
        "CIRCLE_COLOR",'''
content = content.replace('        "SILHOUETTE_COUNT",\n        "SILHOUETTE_COLORS",\n        "SILHOUETTE_DIST",', optional_insert)

extraction_insert = '''    values["silhouette_distances"] = (
        _number_tuple(kind, name, data, "SILHOUETTE_DIST")
        if "SILHOUETTE_DIST" in data
        else None
    )
    values["anim_length"] = _optional_typed(kind, name, data, "ANIM_LENGTH", int, None)
    values["circle_thickness"] = _optional_typed(kind, name, data, "CIRCLE_THICKNESS", int, None)
    if "CIRCLE_COLOR" in data:
        colors = []
        raw_colors = data["CIRCLE_COLOR"]
        if not isinstance(raw_colors, list) or len(raw_colors) != 2:
            raise CatalogValidationError(
                f"Ability '{name}' field 'CIRCLE_COLOR' must be an array of length 2"
            )
        for index, color in enumerate(raw_colors):
            if not isinstance(color, list) or len(color) != 4:
                raise CatalogValidationError(
                    f"Ability '{name}' field 'CIRCLE_COLOR[{index}]' must contain 4 values"
                )
            parsed = tuple(
                _typed(kind, name, f"CIRCLE_COLOR[{index}][{channel}]", item, int)
                for channel, item in enumerate(color)
            )
            if any(channel < 0 or channel > 255 for channel in parsed):
                raise CatalogValidationError(
                    f"Ability '{name}' CIRCLE_COLOR channels must be between 0 and 255"
                )
            colors.append(parsed)
        values["circle_color"] = tuple(colors)
    else:
        values["circle_color"] = None'''
content = content.replace('    values["silhouette_distances"] = (\n        _number_tuple(kind, name, data, "SILHOUETTE_DIST")\n        if "SILHOUETTE_DIST" in data\n        else None\n    )', extraction_insert)

with open('src/Objects/Ships/catalog.py', 'w', encoding='utf-8') as f:
    f.write(content)
