from django.db import models
from django.contrib.auth.models import User

class SimplexProblem(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='problemas_simplex')
    TIPO_CHOICES = [('max', 'Maximizar'), ('min', 'Minimizar')]
    optim = models.CharField(max_length=3, choices=TIPO_CHOICES)
    n = models.PositiveIntegerField()
    m = models.PositiveIntegerField()
    obj = models.CharField(max_length=255)  # Guarda como texto separado por comas
    cons = models.TextField()               # Guarda cada restricción como línea de texto
    types = models.CharField(max_length=255) # Tipos de restricción separados por coma
    rhs = models.CharField(max_length=255)   # RHS separados por coma
    resultado = models.TextField(blank=True, null=True)  # Solución serializada (JSON o texto)
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Problema Símplex #{self.id} ({self.get_optim_display()})"