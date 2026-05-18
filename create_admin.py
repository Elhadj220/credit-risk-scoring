import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings')
django.setup()

from django.contrib.auth.models import User

if not User.objects.filter(username='leuz').exists():
    User.objects.create_superuser('leuz', '', 'Elhadj220')
    print('Superuser created')
else:
    print('Superuser already exists')