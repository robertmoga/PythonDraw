from django.shortcuts import render
from django.views.generic import View
from django.views import generic
from .models import Symbol
from django.http import HttpResponse, HttpResponseRedirect
import json
from django.http import JsonResponse



class IndexView(generic.ListView):
    template_name = 'draw/indexTemp.html'
    context_object_name = 'all_symbols'

    def get_queryset(self):
        return Symbol.objects.all()


def test(request):

    return render(request, 'draw/test.html')

def test2(request):
    info = None
    if request.method == 'GET':
        info = request.GET.get('info')
        print(">> " + str(info))
        info = str(info) + str('hue')

    data = dict()
    data['info'] = info
    #procesare de imagine
    #output imagini pe disc


    return JsonResponse(data, safe=False)

