# SystemMOP_PL

Desarrollo de herramienta en línea de método matemático para optimizar (maximizar o minimizar) una función de objetivo lineal, sujeta a restricciones lineales 

## Descripcion: 
Este proyecto es una herramienta en línea para resolver problemas de programación lineal utilizando el método simplex y métodos gráficos. Permite a los usuarios maximizar o minimizar funciones objetivo sujetas a restricciones lineales.

## Estructura del Proyecto

- **.gitignore**: Archivos y directorios que deben ser ignorados por Git.
- **README.md**: Documentación del proyecto.
- **requirements.txt**: Lista de paquetes de Python requeridos.
- **manage.py**: Utilidad de línea de comandos para interactuar con el proyecto Django.
- **.env**: Variables de entorno para la configuración del proyecto.
- **docker-compose.yml**: Archivo opcional para levantar el entorno con Docker (a Future XD).
- **lineal_solver/**: Configuración principal de Django.
- **apps/**: Contiene las aplicaciones del proyecto.
  - **metodo_simplex/**: Aplicación para resolver problemas con el método simplex.
  - **metodo_grafico/**: Aplicación para resolver problemas cone el metodo grafico.
- **templates/**: Plantillas HTML globales para el proyecto.
- **static/**: Archivos estáticos globales para el proyecto.

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