from django.http import HttpResponse

def hello_world(request):
    print(">>>Sunt in view ")
    return HttpResponse("Hello world <p> <b>this is my app !")

