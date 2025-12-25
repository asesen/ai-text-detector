import fire
from ai_text_detector.config import load_config
from ai_text_detector.gdrive import download_data_from_gdrive
from ai_text_detector.train import train


class Commands:
    def train(self, overrides=None):
        """
        python commands.py train training.training.epochs=3
        """
        cfg = load_config(overrides)
        train(cfg)

    def download_data(self, overrides=None):
        """
        Download dataset from Google Drive.

        Example:
        python commands.py download_data
        python commands.py download_data data.data_dir=data_local
        """
        cfg = load_config(overrides)
        download_data_from_gdrive(cfg)


if __name__ == "__main__":
    fire.Fire(Commands)
