"""
Configuración de ASGI para el proyecto lineal_solver.

Expone el objeto ASGI invocable como una variable a nivel de módulo denominada ``application``.

Para más información sobre este archivo, consulte
https://docs.djangoproject.com/en/stable/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lineal_solver.settings')

application = get_asgi_application()