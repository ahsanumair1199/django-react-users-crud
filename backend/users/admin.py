from django.contrib import admin
from .models import FakeUser


class FakeUserAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email')
    list_display_links = ('id', 'name', 'email')


admin.site.register(FakeUser, FakeUserAdmin)
