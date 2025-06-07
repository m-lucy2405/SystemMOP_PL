from django import forms

class SimplexForm(forms.Form):
    objective_function = forms.CharField(
        label='Función Objetivo',
        max_length=255,
        help_text='Ingrese la función objetivo en términos de las variables.'
    )
    constraints = forms.CharField(
        label='Restricciones',
        widget=forms.Textarea,
        help_text='Ingrese las restricciones, una por línea.'
    )
    variable_names = forms.CharField(
        label='Nombres de Variables',
        max_length=255,
        help_text='Ingrese los nombres de las variables, separados por comas.'
    )
    maximize = forms.BooleanField(
        required=False,
        initial=True,
        label='Maximizar',
        help_text='Marque si desea maximizar la función objetivo.'
    )