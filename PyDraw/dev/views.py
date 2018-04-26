from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import generic
import sys
from django.views.decorators.csrf import csrf_exempt,csrf_protect, requires_csrf_token
from django.core.files.uploadedfile import SimpleUploadedFile
import os
import cv2
import base64



globalData = []
globalData.append("PyDraw Dev 1.0")
NEW_DATASET = 'pd_dataset'


class IndexView(generic.TemplateView):
    template_name = 'dev/index.html'
    sys.stderr.write(">> Dev index accessed \n")


@csrf_exempt
def IndexTemp(request):
    sys.stderr.write(">> TEMP INDEX ACCESSED \n")
    if request.method == 'GET':
        return render(request, 'dev/index.html')
    elif request.method == 'POST':
        data = request.POST.get('data')
        print(">> IMG DATA : " + str(data))
        f = open('F:\\_Licenta\\PyDraw\\PyDraw\\dev\\tempFiles\\fis.txt', 'w')
        f.write(data)
        f.close()

        return HttpResponseRedirect('/')



@csrf_exempt
def testView(request):
    if request.method == 'GET':
        temp = len(globalData)
        context = {'info': globalData, 'len': temp}
        return render(request, 'dev/test.html', context)
    elif request.method == 'POST':

        data = request.POST.get('data')
        print(">> Client msg : " + str(data))
        globalData.append(data)
        return HttpResponseRedirect('/')


@csrf_exempt
def test2View(request):
    temp = len(globalData)
    context = {'info': globalData, 'len': temp}
    return render(request, 'dev/test.html', context)

@csrf_exempt
def BuildDataSet(request):
    if request.method == 'GET':
        context = {'info': 'nothing'}
        return render(request, 'dev/build.html', context)
    elif request.method == 'POST':
        imageData = request.POST.get('imageData')
        imageLabel = request.POST.get('imageLabel')
        print(">>  Data recieved : " + str(imageLabel) + "   " + str(imageData[:21]))


        INDEX_DOC = NEW_DATASET + "/index.txt"
        BUFFER_DOC = NEW_DATASET + "/buff.txt"
        IMAGES_DIR = NEW_DATASET + "/images/"

        # print(">. Current directory : " + str(os.path.dirname(os.path.abspath(__file__)) + BUFFER_DOC))

        # if not os.path.exists(NEW_DATASET):
        #     os.makedirs(NEW_DATASET)
        # if not os.path.exists(IMAGES_DIR):
        #     os.makedirs(IMAGES_DIR)

        #read the last picture
        picNum = -1
        try:
            with open(BUFFER_DOC, 'r') as f:
                temp = f.read()
                if int(temp) > picNum:
                    picNum = int(temp)
                    print(">>> PIC NUM 1 : " + str(picNum))
                f.close()
        except Exception as e:
            print(" >> Eroare la citire din buffer " + str(e))

        test_file(BUFFER_DOC)

        #build name for the new image

        new_image_name = imageLabel+"_"+str(picNum)+".png"
        new_image_path =IMAGES_DIR+new_image_name

        #save object in directory

        save_base64_to_img(imageData, new_image_path)

        #write in index file the name of the image and the representation in ascii

        try:
            new_line = new_image_name+" "+str(ord(imageLabel))
            with open(INDEX_DOC, 'a') as f:
                f.write(str(new_line)+"\n")
                f.close()
        except Exception as e:
            print(">> Eroare la scriere in index : " + str(e))


        #write the new number in the file
        picNum += 1

        try:
            with open(BUFFER_DOC, 'w') as f :
                f.write(str(picNum))
                print(">>> PIC NUM 2 : " + str(picNum))
                f.close()
        except Exception as e:
            print(" >> Eroare la scriere in buffer :" +str(e))

        test_file(BUFFER_DOC)

        return HttpResponseRedirect('/')

#these methods should be moved into another directory
def partition(strOne):
    if ',' in strOne:
        strOne = strOne.partition(",")[2]
        pad = len(strOne) % 4
        strOne += "=" * pad
        return strOne
    else:
        return strOne


def save_base64_to_img(data, path):
    base64_str = partition(data)
    binary_str = base64.b64decode(base64_str)

    with open(path, "wb") as fh:
        fh.write(binary_str)
        fh.close()


def test_file(path):
    with open(path,'r') as f:
        temp = f.read()
        print(">> Test read : " + temp)
        f.close()