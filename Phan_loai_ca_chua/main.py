import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

def read_img(name):
    img = cv2.imread(name)
    if(img.shape[0] > 2000):
        img = cv2.resize(img, dsize = None, fx = 0.25, fy = 0.25)
        return img
    if(img.shape[0] > 1000):
        img = cv2.resize(img, dsize = None, fx = 0.35, fy = 0.35)
        return img
    if(img.shape[0] > 500):
        img = cv2.resize(img, dsize = None, fx = 0.75, fy = 0.75)
        return img
def get_HSV(img, color = "red"): ## default color is red
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = None
    upper = None
    if(color == "red"):
        lower = np.array([0, 100, 100])
        upper = np.array([100, 255, 255])
    if(color == "green"):
        ## have not checked yet
        lower = np.array([0, 50, 0])
        upper = np.array([230, 230, 230])

    if(color == "mix"):
        lower = np.array([0, 50, 0])
        upper = np.array([230, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res
def get_mean(hsv):
    res = [0, 0, 0]
    num = 0
    for row in hsv:
        for pixel in row:
            if(np.mean(pixel) != 0):
                res[0] += pixel[0]
                res[1] += pixel[1]
                res[2] += pixel[2]
                num += 1
    res[0] /= num
    res[1] /= num
    res[2] /= num
    return res
def calculate_threshold(name_of_good, name_of_bad):
    red_tomato = read_img(r"E:\Project Python\Phanbiet\anh2.jpg")
    green_tomato = read_img(r"E:\Project Python\Phanbiet\anh1.jpg")

    red_hsv = get_HSV(red_tomato)
    green_hsv = get_HSV(green_tomato, "green")
    cv2.imshow('anh qua chin', red_hsv)
    cv2.imshow('anh qua non', green_hsv)
    _min = get_mean(green_hsv)   ## G point will be highest
    _max = get_mean(red_hsv)     ## R point will be highest
    ## threshold
    B_threshold = (_min[0] + _max[0]) / 2
    G_threshold = (_min[1] + _max[1]) / 2
    R_threshold = (_min[2] + _max[2]) / 2
    return [B_threshold, G_threshold, R_threshold]
def compare(threshold, _input):
    B_point = _input[0] - threshold[0]
    G_point = _input[1] - threshold[1]
    R_point = _input[2] - threshold[2]
    if(R_point >= 0 and R_point > G_point): return "R"
    if(G_point >= 0 and G_point > R_point): return "G"
    if(G_point == R_point):
        if(B_point >= 0): return "G"
        else: return "R"
    return "-"
### check third image
def decision(name_of_good, name_of_bad, name_to_check):
    print("Plz waitting...")
    file_check = read_img(r"E:\Project Python\Phanbiet\check.jpg")
    hsv = get_HSV(file_check, "mix")
    cv2.imshow('anh can check', file_check)
    threshold = calculate_threshold("anh2.jpg", "anh1.jpg")
    GR_compare = [0, 0]

    for row in hsv:
        for pixel in row:
            if(np.mean(pixel) != 0):
                cp = compare(threshold, pixel)
                if(cp == "R"): GR_compare[1] += 1
                if(cp == "G"): GR_compare[0] += 1
    if(GR_compare[0] >= GR_compare[1]): print("Chua chin!")
    else: print("Da Chin!")
    percent = (GR_compare[1] / (GR_compare[0] + GR_compare[1]))
    print("Da chin : " + str(percent * 100) + "%")
    age = int(percent * 12)
    print("Qua da " + str(age) + " ngay tuoi, con " + str(12 - age) + " ngay nua se chin do va bat dau thu hoach!")
    cv2.waitKey()
    cv2.destroyAllWindows()
decision("anh2.jpg", "anh1.jpg", "check.jpg")
