from django.apps import AppConfig
import pickle
import os


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    # Chargement du modèle au démarrage de l'app — une seule fois
    model = None
    scaler = None
    threshold = 0.35  # seuil optimisé en Semaine 4

    def ready(self):
        model_path  = 'data/processed/model.pkl'
        scaler_path = 'data/processed/dataset_final.pkl'

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                PredictorConfig.model = pickle.load(f)

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                PredictorConfig.scaler = data['scaler']