import os
import cv2
import time

DATASET_NAME = "pd_dataset"
FILE_NAME = "pd_letters.csv"


def resize_specific_dim(img, val):
    resized_image = cv2.resize(img, (val, val))
    # res = cv2.resize(img, None, fx=1/10, fy=1/10, interpolation=cv2.INTER_CUBIC) #for scaling up
    return resized_image


def get_path(dataset_name):
    parent_dir =  os.path.abspath('../..')
    data_path = "\\"+dataset_name+"\\"+"images"
    path = parent_dir + data_path
    return path


def plotData(image, winname):

    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iterate_through_images(path):
    file_name = init_file(path)
    for img in os.listdir(path):
        try:
            line_id = get_ascii(img)
            line = build_line(path, img)
            write_to_file(line_id, line, file_name)

        except Exception as e:
            print("Exception when iterating through images : " + str(e))


def init_file(path):

    path = path.split('\\')
    new_path = str()
    for index_dir in range(len(path)-1):
        new_path += path[index_dir] + "\\"

    file_name = new_path + FILE_NAME
    print(file_name)

    with open(file_name, 'w') as f:
        f.write("")
        f.close()
    return file_name


def get_ascii(img):
    char = img[0]
    return ord(char)


def build_line(path, img_name):
    img_path = path+ "\\"+ img_name
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = resize_specific_dim(img, 100)
    letter = str()

    for line in img:
        for elem in line:
            try:
                letter = letter + str(elem) + ","
            except Exception as e :
                print("Exception in building a new text : " + str(e))

    return letter[0:-1]  # return without the last comma
    # plotData(img, "hue")


def write_to_file(line_id, line, file_name="none"):

    new_text = str(line_id) + "," + str(line)
    # print(new_text)

    with open(file_name, 'a') as f:
        try:
            f.write(new_text)
            f.write("\n")
            f.close()
        except Exception as e:
            print("Execption when writing in cvs : " + str(e))


if __name__ == "__main__":
    timer = time.time()
    ds_path = get_path(DATASET_NAME)
    iterate_through_images(ds_path)

    print("Time elapsed " + str( time.time() - timer ))