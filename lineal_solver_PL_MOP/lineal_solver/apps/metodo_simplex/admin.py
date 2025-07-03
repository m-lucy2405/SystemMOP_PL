from django.contrib import admin
from .models import SimplexProblem

@admin.register(SimplexProblem)
class SimplexProblemAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'optim', 'n', 'm',
        'obj', 'cons', 'types', 'rhs',
        'resultado', 'fecha'
    )
    list_filter = ('optim', 'fecha')
    search_fields = ('id', 'obj', 'resultado')
    readonly_fields = ('fecha',)

    fieldsets = (
        ('Información general', {
            'fields': ('optim', 'n', 'm', 'fecha')
        }),
        ('Función Objetivo', {
            'fields': ('obj',)
        }),
        ('Restricciones', {
            'fields': ('cons', 'types', 'rhs')
        }),
        ('Resultado', {
            'fields': ('resultado',)
        }),
    )
