from django.db import models
from django.contrib.auth.models import User


# class CustomUser(models.Model):
#     name = models.CharField(max_length=64, null=False, unique=True)


class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.PROTECT)
    text = models.TextField(null=False)
    reply = models.IntegerField(null=True)
