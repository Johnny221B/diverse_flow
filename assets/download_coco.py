import os
import subprocess
import zipfile

def download(url, out_path):
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return
    print(f"[download] {url}")
    subprocess.check_call(["wget", "-O", out_path, url])

if __name__ == "__main__":
    root = "coco2017"
    os.makedirs(root, exist_ok=True)

    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    # download
    for fname, url in urls.items():
        out = os.path.join(root, fname)
        download(url, out)

    # unzip
    for fname in urls.keys():
        zip_path = os.path.join(root, fname)
        print(f"[unzip] {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    # optional: remove zips
    for fname in urls.keys():
        os.remove(os.path.join(root, fname))

    print("Done. Layout:")
    print("  coco2017/train2017/")
    print("  coco2017/val2017/")
    print("  coco2017/annotations/")
