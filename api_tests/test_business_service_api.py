import unittest

import requests


class TestBusinessService(unittest.TestCase):

    def test_predict_by_text_image(self):
        payload = {
            "text": "журнальный стол",
            "picture_url": "https://www.tula.bestmebelshop.ru/upload/"
                           "resize_cache/iblock/e48/"
                           "1000_581_1cb3ed90bb4b099ee9292aa8dda3584d7/"
                           "e48c528fd35cf107db9eff73e95bd474.jpg"
        }
        response = requests.post(
            url="http://localhost:8021/predict-by-text-image",
            json=payload,
            timeout=10
        )
        self.assertEqual(response.status_code, 200)

    def test_predict_url(self):
        data = {
            "url": "https://www.tula.bestmebelshop.ru/catalog/"
                   "stoly-zhurnalnye/zhurnalnyy-stol-lava-7-bms/"
        }

        response = requests.post(
            url="http://localhost:8021/predict-by-url",
            json=data,
            timeout=10,
        )
        self.assertEqual(response.status_code, 200)

    def test_predict_invalid_format(self):
        payload = {
            "inputs": {
                "wrong_field": [1, 2, 3]
            }
        }
        response = requests.post(
            url="http://localhost:8021/predict-by-url",
            json=payload,
            timeout=10
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
