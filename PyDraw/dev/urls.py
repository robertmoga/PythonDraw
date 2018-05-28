from django.conf.urls import url
from . import views

app_name = 'dev'

urlpatterns = [
    # #/dev/
    url(r'^old/$', views.IndexTemp),
    url(r'^$', views.IndexView),
    url(r'^process/$', views.IndexProcessing),
    url(r'^test/$', views.testView),
    url(r'^test2/$', views.test2View),
    url(r'^build/$', views.BuildDataSet)
    ]
