from django.urls import path
from . import views

urlpatterns = [
    path('create-users/', views.populate_users, name='create_users'),
]
