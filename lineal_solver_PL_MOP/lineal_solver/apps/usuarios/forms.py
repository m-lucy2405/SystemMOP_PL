from django import forms
from django.contrib.auth import authenticate
from .models import PerfilUsuario
from django.contrib.auth.models import User

class PerfilUsuarioForm(forms.ModelForm):
    first_name = forms.CharField(label='Nombre', max_length=150)
    last_name = forms.CharField(label='Apellido', max_length=150)
    password = forms.CharField(label='Contraseña actual', widget=forms.PasswordInput)

    class Meta:
        model = PerfilUsuario
        fields = ['imagen', 'telefono', 'ciudad', 'fecha_nacimiento', 'direccion']
        widgets = {
            'fecha_nacimiento': forms.DateInput(attrs={'type': 'date'}),
        }

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if self.user:
            self.fields['first_name'].initial = self.user.first_name
            self.fields['last_name'].initial = self.user.last_name

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if not self.user.check_password(password):
            raise forms.ValidationError("La contraseña no es correcta.")
        return password
from django import forms
from django.contrib.auth.models import User

class CambiarUsuarioForm(forms.Form):
    nuevo_username = forms.CharField(
        label="Nuevo nombre de usuario",
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nuevo nombre de usuario'
        })
    )
    password = forms.CharField(
        label="Contraseña actual",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contraseña actual'
        })
    )

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if not self.user.check_password(password):
            raise forms.ValidationError("La contraseña no es correcta.")
        return password

    def clean_nuevo_username(self):
        nuevo_username = self.cleaned_data.get('nuevo_username')
        if User.objects.exclude(pk=self.user.pk).filter(username=nuevo_username).exists():
            raise forms.ValidationError("Este nombre de usuario ya está en uso.")
        return nuevo_username


class CambiarContrasenaForm(forms.Form):
    password_actual = forms.CharField(
        label="Contraseña actual",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contraseña actual'
        })
    )
    nueva_password = forms.CharField(
        label="Nueva contraseña",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nueva contraseña'
        })
    )
    confirmar_password = forms.CharField(
        label="Confirmar nueva contraseña",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirmar nueva contraseña'
        })
    )

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

    def clean_password_actual(self):
        password = self.cleaned_data.get('password_actual')
        if not self.user.check_password(password):
            raise forms.ValidationError("La contraseña actual no es correcta.")
        return password

    def clean(self):
        cleaned_data = super().clean()
        nueva = cleaned_data.get('nueva_password')
        confirmar = cleaned_data.get('confirmar_password')
        if nueva and confirmar and nueva != confirmar:
            self.add_error('confirmar_password', "Las contraseñas no coinciden.")
