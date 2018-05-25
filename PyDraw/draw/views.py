from django.shortcuts import render
from django.views.generic import View
from django.views import generic
from .models import Symbol
from django.http import HttpResponse, HttpResponseRedirect
import json
from django.http import JsonResponse



class IndexView(generic.ListView):
    template_name = 'draw/index.html'
    context_object_name = 'all_symbols'

    def get_queryset(self):
        return Symbol.objects.all()

def home(request):
    pass
    # tmpl_vars = {
    #     'all_posts': Post.objects.reverse(),
    #     'form': PostForm()
    # }
    # return render(request, 'draw/index.html', tmpl_vars)


def create_post(request):
    pass
    # if request.method == 'POST':
    #     post_text = request.POST.get('the_post')
    #     response_data = {}
    #
    #     post = Post(text=post_text, author="Robert")
    #     post.save()
    #
    #     response_data['result'] = 'Create post successful!'
    #     response_data['postpk'] = post.pk
    #     response_data['text'] = post.text
    #     response_data['created'] = post.created.strftime('%B %d, %Y %I:%M %p')
    #     response_data['author'] = "Robert"
    #
    #     # return HttpResponse(
    #     #     json.dumps(response_data),
    #     #     content_type="application/json")
    #     # return HttpResponse("/draw")
    #     return HttpResponseRedirect('/')
    #
    # else:
    #     return HttpResponse(
    #         json.dumps({"nothing to see": "this isn't happening"}),
    #         content_type="application/json"
    #     )


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

