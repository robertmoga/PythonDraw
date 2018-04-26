from django.conf.urls import url
from . import views

app_name = 'dev'

urlpatterns = [
    # #/dev/
    url(r'^$', views.IndexTemp),
    url(r'^test/$', views.testView),
    url(r'^test2/$', views.test2View),
    url(r'^build/$', views.BuildDataSet)
    ]
