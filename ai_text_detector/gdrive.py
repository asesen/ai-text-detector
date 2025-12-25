from pathlib import Path

import gdown


def download_dataset_folder(folder_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not any(output_dir.iterdir()):
        gdown.download_folder(id=folder_id, output=str(output_dir), quiet=False)

    return output_dir
