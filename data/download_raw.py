"""Download the raw files that data.build_main reads, at the revisions the paper's datasets were built from.

  python -m data.download_raw --raw <raw data folder>

`data/sources.json` lists every file that the build reads: where it comes from (a Hugging Face dataset at a commit,
a GitHub repository at a commit, or an official address), its size and its sha256. Each file is written to
<raw>/<owner>__<name>/<path>, the layout that data.build_main expects, and its sha256 is checked. A file that is
already there with the right sha256 is not downloaded again. The tokenizer of Ouro-1.4B, which counts meta.tokens,
is fetched at its commit into <raw>/ByteDance__Ouro-1.4B; pass that folder as --tokenizer.

FOLIO is gated on Hugging Face: accept its terms on https://huggingface.co/datasets/yale-nlp/FOLIO and log in
(`hf auth login`) first. With --check the script only checks the files that are already in --raw.
The script stops with a list of the files that are missing or differ; the build is then not the paper's.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile

SOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sansi-data"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch_hf(repo, path, revision, repo_type, tmp):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type=repo_type, revision=revision, local_dir=tmp)


def fetch(src, tmp):
    """The bytes of one source file (a path to a downloaded file, or bytes)."""
    if "hf_dataset" in src:
        return fetch_hf(src["hf_dataset"], src["path"], src["revision"], "dataset", tmp)
    if "github" in src:
        return fetch_url("https://raw.githubusercontent.com/%s/%s/%s" % (src["github"], src["commit"], src["path"]))
    data = fetch_url(src["url"])
    if "zip_member" in src:
        if hashlib.sha256(data).hexdigest() != src["zip_sha256"]:
            raise ValueError("the archive %s has changed" % src["url"])
        return zipfile.ZipFile(io.BytesIO(data)).read(src["zip_member"])
    return data


def place(got, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if isinstance(got, bytes):
        with open(dest, "wb") as f:
            f.write(got)
    else:
        shutil.copyfile(got, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="the folder to write the raw datasets to")
    ap.add_argument("--check", action="store_true", help="only check the files that are already there")
    args = ap.parse_args()
    S = json.load(open(SOURCES))
    tmp = os.path.join(args.raw, ".download")
    tok = S["tokenizer"]
    jobs = [(os.path.join(tok["folder"], f["path"]), f, {"hf_model": tok["hf_model"], "revision": tok["revision"],
                                                          "path": f["path"]}) for f in tok["files"]]
    jobs += [(f["file"], f, f["source"]) for f in S["files"]]
    bad = []
    for rel, f, src in jobs:
        dest = os.path.join(args.raw, rel)
        if os.path.exists(dest) and sha256(dest) == f["sha256"]:
            print("ok        ", rel)
            continue
        if args.check:
            bad.append((rel, "missing" if not os.path.exists(dest) else "differs"))
            print("MISSING   " if not os.path.exists(dest) else "DIFFERS   ", rel)
            continue
        try:
            if "hf_model" in src:
                got = fetch_hf(src["hf_model"], src["path"], src["revision"], "model", tmp)
            else:
                got = fetch(src, tmp)
            place(got, dest)
        except Exception as e:  # report every failure, then stop
            bad.append((rel, "download failed: %s" % e))
            print("FAILED    ", rel, "-", e)
            continue
        ok = sha256(dest) == f["sha256"]
        print("downloaded" if ok else "DIFFERS   ", rel)
        if not ok:
            bad.append((rel, "the downloaded file differs from the one the paper used"))
    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if bad:
        print("%d of %d files are missing or differ:" % (len(bad), len(jobs)))
        for rel, why in bad:
            print("  %s: %s" % (rel, why))
        sys.exit(1)
    print("all %d files match the ones the paper's datasets were built from" % len(jobs))
    print("tokenizer folder for --tokenizer:", os.path.join(args.raw, tok["folder"]))


if __name__ == "__main__":
    main()
