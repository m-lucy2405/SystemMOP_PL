from django.db import models
from django.contrib.auth.models import User
class ProblemaGrafico(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='problemas_grafico', null=True, blank=True)
    nombre = models.CharField(max_length=100)
    restricciones = models.TextField(help_text="JSON serializado de restricciones")
    funcion_objetivo = models.CharField(max_length=255)
    solucion_optima = models.JSONField(null=True, blank=True)
    vertices_factibles = models.JSONField(null=True, blank=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.nombre
