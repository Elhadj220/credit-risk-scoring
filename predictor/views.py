from django.shortcuts import render

# Create your views here.
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.apps import apps

EXPECTED_FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT_MEAN', 'BILL_AMT_TREND', 'BILL_AMT_MAX',
    'PAY_AMT_MEAN', 'PAY_RATIO'
]


class PredictView(APIView):
    permission_classes = [IsAuthenticated]  # ← JWT requis

    def post(self, request):
        config = apps.get_app_config('predictor')

        if config.model is None:
            return Response(
                {'error': 'Modèle non chargé'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        data = request.data
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return Response(
                {'error': f'Features manquantes : {missing}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            features = np.array([[data[f] for f in EXPECTED_FEATURES]])
            features_scaled = config.scaler.transform(features)
        except Exception as e:
            return Response(
                {'error': f'Erreur de traitement : {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        risk_score = config.model.predict_proba(features_scaled)[0][1]
        prediction = 'défaut' if risk_score >= config.threshold else 'sain'

        return Response({
            'risk_score': round(float(risk_score), 4),
            'prediction': prediction,
            'threshold':  config.threshold,
            'risk_level': 'élevé' if risk_score > 0.6 else
                         'moyen' if risk_score > 0.3 else 'faible',
            'requested_by': request.user.username  # ← log utilisateur
        })


class HealthView(APIView):
    permission_classes = [AllowAny]  # ← health check public

    def get(self, request):
        config = apps.get_app_config('predictor')
        return Response({
            'status': 'ok',
            'model_loaded': config.model is not None,
            'threshold': config.threshold
        })