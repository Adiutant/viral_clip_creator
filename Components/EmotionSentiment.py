import torch
from transformers import BertForSequenceClassification, AutoTokenizer


class EmotionSentiment:
    def __init__(self):
        self.LABELS = {'neutral': 0, 'happiness': 5, 'sadness': 2, 'enthusiasm': 4, 'fear': 4, 'anger': 4, 'disgust': 3}
        self.SCORE = [0, 5, 2, 4, 4, 4, 3]
        self.tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
        self.model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')

    def predict_emotion(self, text: str) -> int:
        """
            Мы принимаем входной текст, разбиваем его на токены, пропускаем через модель и возвращаем прогнозируемую метку.
            :param text: Текст для классификации
            :type text: str
            :return: Оценка по шкале для прогнозируемой эмоции
        """
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()

        return self.SCORE[predicted[0]]
