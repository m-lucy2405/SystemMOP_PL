# SystemMOP_PL

Desarrollo de herramienta en línea de método matemático para optimizar (maximizar o minimizar) una función de objetivo lineal, sujeta a restricciones lineales 

## Descripcion: 
Este proyecto es una herramienta en línea para resolver problemas de programación lineal utilizando el método simplex y métodos gráficos. Permite a los usuarios maximizar o minimizar funciones objetivo sujetas a restricciones lineales.

## Estructura del Proyecto

```
lineal_solver/
├── .gitignore                  # Archivos y carpetas ignorados por Git
├── README.md                   # Documentación del proyecto
├── requirements.txt            # Lista de paquetes de Python requeridos
├── manage.py                   # Utilidad de línea de comandos para Django
├── .env                        # Variables de entorno para la configuración
├── docker-compose.yml          # (Opcional) Para levantar el entorno con Docker
├── lineal_solver/              # Configuración principal de Django
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── apps/                       # Contiene las aplicaciones del proyecto
│   ├── metodo_simplex/         # Aplicación para resolver con método símplex
│   │   ├── migrations/
│   │   │   └── __init__.py
│   │   ├── templates/
│   │   ├── static/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── forms.py
│   │   └── simplex_solver.py
│   └── metodo_grafico/         # Aplicación para resolver gráficamente
│       ├── migrations/
│       │   └── __init__.py
│       ├── templates/
│       ├── static/
│       ├── models.py
│       ├── views.py
│       ├── urls.py
│       ├── forms.py
│       └── grafico_solver.py
├── templates/                  # Plantillas HTML globales para el proyecto
│   └── base.html
├── static/                     # Archivos estáticos globales
│   ├── css/
│   └── js/
```

## Instalación

1. Clona el repositorio:
   ```
   git clone <URL del repositorio>
   cd lineal_solver
   ```

2. Crea un entorno virtual y actívalo:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura las variables de entorno en el archivo `.env`.

5. Ejecuta las migraciones:
   ```
   python manage.py migrate
   ```

6. Inicia el servidor de desarrollo:
   ```
   python manage.py runserver
   ```

## Uso

Accede a la aplicación en tu navegador en `http://127.0.0.1:8000/`. Desde allí, podrás utilizar las funcionalidades para resolver problemas de programación lineal.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para discutir cambios o mejoras.