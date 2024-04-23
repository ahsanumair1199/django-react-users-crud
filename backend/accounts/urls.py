from django.urls import path
from . import views, serializers

urlpatterns = [
    path('login/', serializers.MyTokenObtainPairView.as_view(),
         name='token_obtain_pair'),
    path('register/', views.register, name='register'),
]
