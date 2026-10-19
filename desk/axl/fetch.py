"""Fetch the original files behind chibifire/datasource-articulation-xl2-parquet's anime/avatar rows
(meta.parquet file_identifier, GitHub blob -> raw), resumable by uuid. One JSONL line per request in
raw/requests.jsonl (uuid, url, http status, bytes, sha256, error); files land in raw/files/<uuid>.<ext>.
A 404/403 is final; timeouts and 429/5xx are left unlogged so a rerun retries them."""
import concurrent.futures as cf
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "raw" / "files"
LOG = ROOT / "raw" / "requests.jsonl"
MAX = 100_000_000


def raw_url(u):
    return urllib.parse.quote(u.replace("github.com", "raw.githubusercontent.com", 1).replace("/blob/", "/", 1), safe=":/")


def get(row):
    url = raw_url(row.file_identifier)
    dest = OUT / ("%s.%s" % (row.uuid, row.file_type))
    rec = dict(uuid=row.uuid, url=url, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "chibifire-axl-eyes"}), timeout=120) as r:
                b = r.read(MAX + 1)
            if len(b) > MAX:
                return dict(rec, status=200, bytes=len(b), error="over %d bytes, not kept" % MAX)
            dest.write_bytes(b)
            return dict(rec, status=200, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                        lfs_pointer=b[:40].startswith(b"version https://git-lfs"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (attempt + 1) * float(e.headers.get("Retry-After", 1) or 1))
                continue
            return dict(rec, status=e.code, error=str(e))
        except Exception as e:  # timeout / reset: transient, retried by a rerun
            if attempt == 4:
                return None
            time.sleep(2 ** (attempt + 1))
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ROOT / "anime_rows.csv")
    done = set()
    if LOG.exists():
        done = {json.loads(l)["uuid"] for l in LOG.open()}
    todo = [r for r in rows.itertuples() if r.uuid not in done]
    print("rows %d, done %d, to fetch %d" % (len(rows), len(done), len(todo)), flush=True)
    n = 0
    with cf.ThreadPoolExecutor(4) as ex, LOG.open("a") as log:
        for rec in ex.map(get, todo):
            n += 1
            if rec:
                log.write(json.dumps(rec) + "\n"); log.flush()
            if n % 100 == 0:
                print("fetched %d / %d" % (n, len(todo)), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
