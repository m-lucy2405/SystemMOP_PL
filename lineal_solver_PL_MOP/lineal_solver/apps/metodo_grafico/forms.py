from django import forms
from .models import ProblemaGrafico

class ProblemaGraficoForm(forms.ModelForm):
    class Meta:
        model = ProblemaGrafico
        fields = ['nombre', 'restricciones', 'funcion_objetivo']
        widgets = {
            'restricciones': forms.Textarea(attrs={'rows': 4, 'placeholder': 'Ej: [{"lhs":[1,2],"sign":"<=","rhs":10}, ...]'}),
            'funcion_objetivo': forms.TextInput(attrs={'placeholder': 'Ej: [3,5]'}),
        }
