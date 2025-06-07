from django import forms

class GraphicalSolverForm(forms.Form):
    objective_function = forms.CharField(
        label='Función Objetivo',
        max_length=255,
        widget=forms.TextInput(attrs={'placeholder': 'Ejemplo: 3x + 4y'})
    )
    constraints = forms.CharField(
        label='Restricciones',
        widget=forms.Textarea(attrs={'placeholder': 'Ejemplo: x + y <= 10\n2x + y <= 15'})
    )
    variable_x = forms.FloatField(
        label='Valor de x',
        required=False
    )
    variable_y = forms.FloatField(
        label='Valor de y',
        required=False
    )