from django.db import models
from django.contrib.auth.models import User

class Symbol(models.Model):
    user = models.CharField(max_length=200)
    file_path = models.CharField(max_length=1000)
    predicted = models.CharField(max_length=25)

    def __str__(self):
        return "[ " + self.user + ", " + self.predicted + " ]"

class Post(models.Model):
    author = models.TextField()
    text = models.TextField()

    # Time is a rhinocerous
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']

    def __unicode__(self):
        return self.text+' - '+self.author.username