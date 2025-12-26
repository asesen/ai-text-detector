from typing import List, Union

import joblib
import numpy as np
import onnxruntime as ort


class TextAIClassifier:
    """
    TF-IDF → ONNX-модель → предсказание 0/1
    """

    def __init__(self, cfg):
        # Загружаем TF-IDF
        tfidf_path = cfg.export.tfidf.output_path
        self.vectorizer = joblib.load(tfidf_path)
        print(f"TF-IDF загружен из: {tfidf_path}")

        providers = ort.get_available_providers()
        onnx_path = cfg.export.onnx.output_path

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(
            f"ONNX-модель загружена: {onnx_path} (providers: {self.session.get_providers()})"
        )

        # Проверяем имя входа (должно быть "input" по твоему export-коду)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Предсказывает класс для одного текста или списка текстов

        Args:
            texts: строка или список строк

        Returns:
            np.ndarray: массив 0/1 (классы)
        """
        # Приводим к списку
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        # TF-IDF → sparse → dense float32
        X = self.vectorizer.transform(texts).astype(np.float32)

        # ONNX ожидает dense массив
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Инференс
        outputs = self.session.run([self.output_name], {self.input_name: X})[0]

        probs = 1 / (1 + np.exp(-outputs))  # sigmoid
        preds = (probs >= 0.5).astype(np.int64).flatten()
        return preds

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Предсказывает вероятности для одного текста или списка текстов

        Args:
            texts: строка или список строк

        Returns:
            np.ndarray: массив 0/1 (классы)
        """
        # Приводим к списку
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        # TF-IDF → sparse → dense float32
        X = self.vectorizer.transform(texts).astype(np.float32)

        # ONNX ожидает dense массив
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Инференс
        outputs = self.session.run([self.output_name], {self.input_name: X})[0]

        probs = 1 / (1 + np.exp(-outputs))  # sigmoid)
        return probs
