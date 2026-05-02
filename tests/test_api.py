import pytest
import json
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status


VALID_PAYLOAD = {
    "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2,
    "MARRIAGE": 2, "AGE": 35, "PAY_0": -1, "PAY_2": -1,
    "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT_MEAN": 10000, "BILL_AMT_TREND": -1000,
    "BILL_AMT_MAX": 20000, "PAY_AMT_MEAN": 5000,
    "PAY_RATIO": 0.5
}


class PredictAPITest(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpass123'
        )

    def get_token(self):
        response = self.client.post('/api/token/', {
            'username': 'testuser',
            'password': 'testpass123'
        }, format='json')
        return response.data['access']

    def test_predict_without_auth_returns_401(self):
        response = self.client.post('/api/predict/', VALID_PAYLOAD, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_predict_with_auth_returns_200(self):
        token = self.get_token()
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = self.client.post('/api/predict/', VALID_PAYLOAD, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('risk_score', response.data)
        self.assertIn('prediction', response.data)

    def test_predict_missing_features_returns_400(self):
        token = self.get_token()
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = self.client.post('/api/predict/', {"LIMIT_BAL": 50000}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_health_endpoint_public(self):
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'ok')

    def test_risk_score_between_0_and_1(self):
        token = self.get_token()
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = self.client.post('/api/predict/', VALID_PAYLOAD, format='json')
        self.assertGreaterEqual(response.data['risk_score'], 0)
        self.assertLessEqual(response.data['risk_score'], 1)