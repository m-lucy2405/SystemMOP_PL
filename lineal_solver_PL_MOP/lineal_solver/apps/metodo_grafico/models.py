from django.db import models
from django.contrib.auth.models import User

class ProblemaGrafico(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='problemas_grafico')
    TIPO_CHOICES = [('max', 'Maximizar'), ('min', 'Minimizar')]
    optim = models.CharField(max_length=3, choices=TIPO_CHOICES)
    n = models.PositiveIntegerField()
    m = models.PositiveIntegerField()
    obj = models.CharField(max_length=100)
    cons = models.TextField(default="")
    types = models.CharField(max_length=100)
    rhs = models.CharField(max_length=100)
    resultado = models.TextField(blank=True, null=True)  # Para guardar el resultado si lo deseas
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Problema {self.id} ({self.get_optim_display()})"