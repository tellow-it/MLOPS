import unittest
import requests


class TestModelService(unittest.TestCase):

    def test_health_success(self):
        response = requests.get("http://localhost:8020/health")
        self.assertEqual(response.status_code, 200)

    def test_predict_single(self):
        payload = {
            "text": "Мебель Мебель Мебельный шкаф Ящик"
        }
        response = requests.post("http://localhost:8020/predict", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_predict_batch(self):
        payload = [
            {
                "text": "Мебель Мебель Мебельный шкаф Ящик"
            },
            {
                "text": "Автомобили Шины Шины Зеркало"
            }
        ]

        response = requests.post("http://localhost:8020/predict", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_predict_invalid_format(self):
        payload = {
            "inputs": {
                "wrong_field": [1, 2, 3]
            }
        }
        response = requests.post("http://localhost:8020/predict", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
