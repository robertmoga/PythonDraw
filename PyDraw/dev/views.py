import base64
import os
import sys

from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from keras import backend as K


from .InputProcessing import DataToImage, InputFormatting, build_name_current
from .intelli_recognition import OutputBuilder, CharClassifier

print("Se apeleaza serverul")
# clf = CharClassifier("dev/tempFiles/keras_pd2_v4.h5")
globalData = []
globalData.append("PyDraw Dev 1.0")
NEW_DATASET = 'pd_dataset'


def IndexView(request):
    return render(request, 'dev/indexTemp.html')

def IndexProcessing(request):
    imgData = None
    if request.method == 'GET':
        imgData = request.GET.get('info')

    # name generator
    name = build_name_current()
    if name is None:
        name = 'help'
    path = str(os.getcwd()) + '\\dev\\temporary\\' + name
    in_form = InputFormatting(path, imgData)
    data_reader = DataToImage(path)
    out_build = OutputBuilder(path)
    out_build.build_images()
    # try :
    clf = CharClassifier("dev/tempFiles/keras_pd2_v4.h5")
    res = out_build.build_output_dict(clf)
    K.clear_session()

    # except Exception as e:
    #     print(" >> Eroare in view la clasificare "  + str(e))
    #     res = out_build.build_output_dict(None)

    return JsonResponse(res, safe=False)



@csrf_exempt
def IndexTemp(request):
    sys.stderr.write(">> TEMP INDEX ACCESSED \n")
    if request.method == 'GET':
        return render(request, 'dev/indexOldBackup.html')
    elif request.method == 'POST':
        data = request.POST.get('data')
        print(">> IMG DATA : " + str(data[:10]))

        try:
            f = open('dev/tempFiles/fis.txt', 'w')
            f.write(data)
            f.close()
        except Exception as e:
            print(">> Eroare la scriere in buffer din /dev : " + str(e))
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
def BuildDataSet(request):
    if request.method == 'GET':
        context = {'info': 'nothing'}
        return render(request, 'dev/build.html', context)
    elif request.method == 'POST':
        imageData = request.POST.get('imageData')
        imageLabel = request.POST.get('imageLabel')
        print(">>  Data recieved : " + str(imageLabel) + "   " + str(imageData[:21]))
        save_images_dataset(imageData, imageLabel)

        return HttpResponseRedirect('/')

@csrf_exempt
def test2View(request):
    temp = len(globalData)
    context = {'info': globalData, 'len': temp}
    return render(request, 'dev/test.html', context)

# method that writes into files and keeps track of the dataset
def save_images_dataset(imageData, imageLabel):
    INDEX_DOC = NEW_DATASET + "/index.txt"
    BUFFER_DOC = NEW_DATASET + "/buff.txt"
    IMAGES_DIR = NEW_DATASET + "/images/"

    # print(">. Current directory : " + str(os.path.dirname(os.path.abspath(__file__)) + BUFFER_DOC))

    # if not os.path.exists(NEW_DATASET):
    #     os.makedirs(NEW_DATASET)
    # if not os.path.exists(IMAGES_DIR):
    #     os.makedirs(IMAGES_DIR)

    # read the last picture
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

    # build name for the new image

    new_image_name = imageLabel + "_" + str(picNum) + ".png"
    new_image_path = IMAGES_DIR + new_image_name

    # save object in directory

    save_base64_to_img(imageData, new_image_path)

    # write in index file the name of the image and the representation in ascii
    try:
        new_line = new_image_name + " " + str(ord(imageLabel))
        with open(INDEX_DOC, 'a') as f:
            f.write(str(new_line) + "\n")
            f.close()
    except Exception as e:
        print(">> Eroare la scriere in index : " + str(e))

    # write the new number in the file
    picNum += 1

    try:
        with open(BUFFER_DOC, 'w') as f:
            f.write(str(picNum))
            print(">>> PIC NUM 2 : " + str(picNum))
            f.close()
    except Exception as e:
        print(" >> Eroare la scriere in buffer :" + str(e))

# these methods should be moved into another directory
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


