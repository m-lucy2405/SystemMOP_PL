from django import forms
from .models import ProblemaGrafico

class ProblemaGraficoForm(forms.ModelForm):
    class Meta:
        model = ProblemaGrafico
        fields = ['nombre', 'restricciones', 'funcion_objetivo']
        widgets = {
            'nombre': forms.TextInput(attrs={'placeholder': 'Nombre descriptivo'}),
            'restricciones': forms.Textarea(attrs={'rows': 4, 'placeholder': '[{"lhs":[1,2],"sign":"<=","rhs":10}, ...]'}),
            'funcion_objetivo': forms.TextInput(attrs={'placeholder': '[c1, c2]'}),
        }
