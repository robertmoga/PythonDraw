from django.conf.urls import url
from . import views

app_name = 'draw'

urlpatterns = [
    # #/draw/
    url(r'^$', views.IndexView.as_view(), name='index'),
    ]
