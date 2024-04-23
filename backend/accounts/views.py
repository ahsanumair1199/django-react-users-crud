from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render, redirect
from .decorators import validate_registration_data
from .models import Account


@api_view(['POST'])
@validate_registration_data
def register(request):
    data = request.data
    user = Account.objects.create(
        first_name=data['first_name'],
        last_name=data['last_name'],
        email=data['email'],
        username=data['email'],
        is_active=True
    )
    user.set_password(data['password'])
    user.save()
    return Response("User created.")
