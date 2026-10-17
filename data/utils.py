import hashlib
import pathlib
import shutil
import urllib.request


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url, path, checksum=None):
    """Download `url` to `path` if the file does not exist yet. Check the MD5 if one is given."""
    path = pathlib.Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"Downloading {url} to {part} ...")
    with urllib.request.urlopen(url) as response, open(part, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 24)
    if checksum is not None and md5(part) != checksum:
        raise RuntimeError(f"{part} is corrupt (MD5 mismatch); delete it and retry.")
    part.rename(path)
