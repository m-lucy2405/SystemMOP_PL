from django import forms

class GraficoForm(forms.Form):
    TIPO_CHOICES = [('max', 'Maximizar'), ('min', 'Minimizar')]
    optim = forms.ChoiceField(choices=TIPO_CHOICES, label="Tipo de optimización")
    n = forms.IntegerField(label="Número de variables", min_value=2, max_value=2, initial=2)
    m = forms.IntegerField(label="Número de restricciones", min_value=1, max_value=10)
    obj = forms.CharField(label="Coeficientes de la función objetivo (separados por coma)")
    cons = forms.CharField(label="Coeficientes de las restricciones (una restricción por línea, coeficientes separados por coma)")
    types = forms.CharField(label="Tipos de restricción (<=, >=, =) separados por coma")
    rhs = forms.CharField(label="Lados derechos (RHS) de las restricciones, separados por coma")