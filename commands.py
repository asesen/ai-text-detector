import fire
from ai_text_detector.config import load_config
from ai_text_detector.train import train


class Commands:
    def train(self, overrides=None):
        """
        python commands.py train training.training.epochs=3
        """
        cfg = load_config(overrides)
        train(cfg)


if __name__ == "__main__":
    fire.Fire(Commands)
