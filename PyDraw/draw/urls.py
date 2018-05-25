from django.conf.urls import url
from . import views

app_name = 'draw'

urlpatterns = [
    # #/draw/
    # url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^$', views.home),
    url(r'^draw/create_post/$', views.create_post),
    url(r'^test/$', views.test),
    url(r'^test2/$', views.test2),

    ]
