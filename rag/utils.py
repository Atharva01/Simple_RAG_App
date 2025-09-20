from pathlib import Path


def save_uploaded_files(uploaded_files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for up in uploaded_files:
        p = dest_dir / up.name
        with open(p, "wb") as f:
            f.write(up.read())
        paths.append(p)
    return paths
