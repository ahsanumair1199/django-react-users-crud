from rest_framework import serializers
from .models import FakeUser


class UserSerializer(serializers.Serializer):
    class Meta:
        model = FakeUser
        fields = '__all__'
