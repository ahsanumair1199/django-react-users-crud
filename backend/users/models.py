from django.db import models
from django.contrib.postgres.fields import ArrayField
from accounts.models import Account



class FakeUser(models.Model):
    account = models.ForeignKey(Account, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    email = models.EmailField(max_length=100, unique=False)
    addresses = ArrayField(models.JSONField(), null=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.email