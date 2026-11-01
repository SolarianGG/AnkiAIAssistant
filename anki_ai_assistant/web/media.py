"""Media files before they go into the collection: real files, not web pages, and images of card size."""

from io import BytesIO
from pathlib import PurePath

# Enough for a card on a phone or a desktop; larger originals only slow down the sync to the phone
MAX_IMAGE_SIDE = 800
# A smaller image that needs no resizing is kept as is
KEEP_IMAGE_BYTES = 300 * 1024
JPEG_QUALITY = 85
# The extension a kept file gets for its real format: models name a JPEG 'x.png'
EXTENSIONS = {"JPEG": (".jpg", ".jpeg"), "PNG": (".png",), "WEBP": (".webp",)}


def check_not_web_page(data: bytes) -> None:
    head = data[:512].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        raise ValueError("The URL returned a web page, not a media file: use a direct link to the file.")


def prepare_image(data: bytes, name: str) -> tuple[bytes, str]:
    """Shrink an image to MAX_IMAGE_SIDE and re-encode it: JPEG, or PNG if it has transparency.
    Returns the data and the file name, whose extension may change. SVG and GIF (may be animated) stay as they are."""
    from PIL import Image, UnidentifiedImageError

    check_not_web_page(data)
    path = PurePath(name)
    if path.suffix.lower() in (".svg", ".gif"):
        return data, name
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"The file is not an image Anki can show ({e}).") from e

    resize = max(image.size) > MAX_IMAGE_SIDE
    if not resize and (image.format == "JPEG" or len(data) <= KEEP_IMAGE_BYTES):
        extensions = EXTENSIONS.get(image.format)
        if extensions and path.suffix.lower() not in extensions:
            name = path.stem + extensions[0]
        return data, name
    alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = BytesIO()
    if alpha:
        image.save(out, "PNG", optimize=True)
        suffix = ".png"
    else:
        image.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
        suffix = ".jpg"
    if not resize and out.tell() >= len(data):
        return data, name
    return out.getvalue(), path.stem + suffix
