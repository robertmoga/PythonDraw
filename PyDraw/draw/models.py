from django.db import models


class Symbol(models.Model):
    user = models.CharField(max_length=200)
    file_path = models.CharField(max_length=1000)
    predicted = models.CharField(max_length=25)

    def __str__(self):
        return "[ " + self.user + ", " + self.predicted + " ]"

