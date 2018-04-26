from django.shortcuts import render
from django.views.generic import View
from django.views import generic
from .models import Symbol


class IndexView(generic.ListView):
    template_name = 'draw/index.html'
    context_object_name = 'all_symbols'

    def get_queryset(self):
        return Symbol.objects.all()

