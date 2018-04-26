import base64
import codecs

'''This script collects base64 data from fis and turns it into .png'''

def read_data():
    f = open('fis.txt', 'r')
    strOne = f.read()
    f.close()
    print(strOne)
    data = partition(strOne)
    print(strOne)
    return data

def partition(strOne):
    if ',' in strOne :
        strOne = strOne.partition(",")[2]
        pad = len(strOne) % 4
        strOne += "=" * pad
        return strOne
    else:
        return strOne


a = 'eW91ciB0ZXh0'

data = read_data()
data = partition(data)
data = base64.b64decode(data)
print(data)

with open("imageToSave.png", "wb") as fh:
    fh.write(data)
