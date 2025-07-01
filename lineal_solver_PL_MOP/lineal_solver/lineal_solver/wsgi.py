"""
Configuración de WSGI para el proyecto lineal_solver.

Expone el objeto invocable de WSGI como una variable a nivel de módulo denominada ``application``.

Para más información sobre este archivo, consulte
https://docs.djangoproject.com/en/stable/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lineal_solver.settings')

application = get_wsgi_application()
