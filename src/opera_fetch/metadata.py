"""What a burst's identity is, and how it lands on the stack it describes.

RTC keeps it in GeoTIFF tags and CSLC in an HDF5 group, so each reader normalises its own
into one shape and this puts it on the Dataset. Written once so the two products cannot
drift into describing themselves differently.
"""

from opera_fetch import constants as const

FIELDS = ("burst_id", "track", "direction", "footprint", "product_version")


def describe(stack, product, identity, granules):
    """Carry the burst's identity onto the stack it came from."""
    missing = [field for field in ("burst_id", "track", "direction") if not identity.get(field)]
    if missing:
        raise ValueError(f"this {product} granule names no {', '.join(missing)}")

    stack.attrs = {
        "product": product,
        "burst_id": str(identity["burst_id"]).upper().replace("_", "-"),
        "track": int(identity["track"]),
        "direction": str(identity["direction"]).upper(),
        "spacing": const.SPACING[product],
        # The outline of the data itself. A burst is a rotated parallelogram and its
        # bounding box is a third larger, so a box overlapping an area proves nothing.
        "footprint": identity.get("footprint") or "",
        # Which granules these values came from. A reprocessing changes the numbers, so
        # without the IDs a saved stack cannot be told apart from one built next year.
        "granules": "\n".join(sorted(granules)),
        # Text, not a number: as a float, v1.10 and v1.1 are the same version.
        "product_version": str(identity.get("product_version") or ""),
    }
    return stack
