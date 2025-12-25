from pathlib import Path

import gdown
from omegaconf import DictConfig


def download_data_from_gdrive(cfg: DictConfig) -> None:
    """
    Download files from Google Drive using file IDs.
    """

    data_cfg = cfg.data
    data_dir = Path(data_cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for split, meta in data_cfg.gdrive_files.items():
        file_id = meta.id
        filename = meta.name
        output_path = data_dir / filename

        if output_path.exists():
            print(f"[SKIP] {filename} already exists")
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[DOWNLOAD] {split}: {filename}")

        gdown.download(url=url, output=str(output_path), quiet=False)
