from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from .Serializers import UserSerializer
from .models import FakeUser
import faker
# END IMPORTS

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def populate_users(request):
    fake = faker.Faker()
    users = []
    for _ in range(10001):
        name = fake.name()
        email = fake.email()
        address = {
            "city": fake.city(),
            "state": fake.state(),
            "country": fake.country(),
            "address_line_1": fake.street_address(),
            "address_line_2": fake.secondary_address(),
            "role": fake.job(),
            "phone": fake.phone_number()
        }

        new_user = FakeUser.objects.create(
            account=request.user,
            name=name,
            email=email,
            addresses=[address]
        )
        new_user.save()
    
    return Response("Users created")