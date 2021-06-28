import random

import cv2
import numpy as np


def get_canny(img_in, min_t, max_t, g):
    kernel = np.ones((2, 2), np.uint8)
    img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    img_out = cv2.blur(img_out, (g, g), 0)
    img_out = cv2.Canny(img_out, min_t, max_t)
    img_out = cv2.dilate(img_out, kernel, iterations=2)
    img_out = cv2.erode(img_out, kernel, iterations=1)
    return img_out


def pro_blue(img_in):
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([94, 174, 128])
    upper = np.array([115, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_blue = cv2.bitwise_and(img_in, img_in, mask=mask)
    img_out = get_canny(img_blue, 20, 40, 5)
    return img_out


def pro_red(img_in):
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([163, 133, 156])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_red = cv2.bitwise_and(img_in, img_in, mask=mask)
    img_out = get_canny(img_red, 20, 40, 5)
    return img_out


def pro_green(img_in):
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([56, 106, 100])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_green = cv2.bitwise_and(img_in, img_in, mask=mask)
    img_out = get_canny(img_green, 20, 40, 5)
    return img_out


def pro_yellow(img_in):
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([94, 174, 128])
    upper = np.array([115, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_yellow = cv2.bitwise_and(img_in, img_in, mask=mask)
    img_out = get_canny(img_yellow, 20, 40, 5)
    return img_out


def pro_orange(img_in):
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower = np.array([3, 165, 180])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_orange = cv2.bitwise_and(img_in, img_in, mask=mask)
    img_out = get_canny(img_orange, 20, 40, 5)
    return img_out


def pro_white(img_in):
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 0, 230])
    upper = np.array([168, 50, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_white = cv2.bitwise_and(img_in, img_in, mask=mask)
    img_out = get_canny(img_white, 20, 40, 5)

    return img_out


def get_contours(img_in):
    contours, hierarchy = cv2.findContours(img_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes = []
    approx = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 7000:

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)

            x, y, w, h = cv2.boundingRect(approx)
            approx = np.array([[[x, y]], [[x + w, y]], [[x, y + h]], [[x + w, y + h]]])
            boxes.append(approx.tolist())
    return boxes


def resquare(squares_in):
    reduced = []
    for color_results in squares_in:
        for square in color_results:
            if reduced:
                reduced = reduced + square
            else:
                reduced = square

    squares_out = []
    for i in range(9):
        square = []
        for j in range(4):
            square.append(reduced[i*4+j])
        squares_out.append(square)

    return squares_out


def reorder_points(points_list):
    points = np.array(points_list)
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    sum_points = points.sum(1)
    new_points[0] = points[np.argmin(sum_points)]
    new_points[3] = points[np.argmax(sum_points)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points.tolist()


def sort_key1(e):
  return e[1]


def sort_key2(e):
  return e[0]


def reorder_squares(squares_list):
    list_out = squares_list.copy()

    indexes = list(range(0, 9))
    centers = []
    for sq in squares_list:
        centers.append([sq[0][0][0] + (sq[3][0][0] - sq[0][0][0]) / 2, sq[0][0][1] + (sq[3][0][1] - sq[0][0][1])])
    centers_sort = centers.copy()
    centers_sort.sort(key=sort_key1)

    section1 = centers_sort[0:3]
    section1.sort(key=sort_key2)
    section2 = centers_sort[3:6]
    section2.sort(key=sort_key2)
    section3 = centers_sort[6:9]
    section3.sort(key=sort_key2)
    centers_sort = section1 + section2 + section3

    for i in range(9):
        for j in range(9):
            if centers_sort[i][0] == centers[j][0] and centers_sort[i][1] == centers[j][1]:
                indexes[i] = j

    for i in range(9):
        list_out[i] = squares_list[indexes[i]]
    return list_out


def get_colors(squares_in, hsv_img):
    list_out = []
    # Blanco = F
    # Azul = U
    # Verde = D
    # Anaranjado = L
    # Rojo = R
    # Amarillo = B
    val = 255
    index = 0
    for sq in squares_in:
        min_0 = sq[0][0][0] + 3
        max_0 = sq[3][0][0] - 3
        min_1 = sq[0][0][1] + 3
        max_1 = sq[3][0][1] - 3

        #cv2.rectangle(img, (min_0, min_1), (max_0, max_1), (0, val, 0), 1)
        val -= 25
        avg_h = 0
        avg_s = 0
        if(index != 4):
            for i in range(10):
                avg_h += hsv_img[random.randrange(min_1, max_1)][random.randrange(min_0, max_0)][0]
                avg_s += hsv_img[random.randrange(min_1, max_1)][random.randrange(min_0, max_0)][1]
            avg_h = avg_h / 10
            avg_s = avg_s / 10
        else:
            for i in range(10):
                avg_h += hsv_img[min_1][min_0][0]
                avg_h += hsv_img[max_1][min_0][0]
                avg_h += hsv_img[min_1][max_0][0]
                avg_h += hsv_img[max_1][max_0][0]
                avg_s += hsv_img[min_1][min_0][1]
                avg_s += hsv_img[max_1][min_0][1]
                avg_s += hsv_img[min_1][max_0][1]
                avg_s += hsv_img[max_1][max_0][1]
                avg_h = avg_h / 4
                avg_s = avg_s / 4
        if avg_s < 100:
            list_out.append('F')
        elif avg_h < 20:
            list_out.append('L')
        elif avg_h < 45:
            list_out.append('B')
        elif avg_h < 73:
            list_out.append('D')
        elif avg_h < 138:
            list_out.append('U')
        else:
            list_out.append('R')

        index += 1
    return list_out


images = [cv2.imread("Resources/F.jpg"), cv2.imread("Resources/U.jpg"), cv2.imread("Resources/D.jpg"), cv2.imread("Resources/L.jpg"), cv2.imread("Resources/R.jpg"), cv2.imread("Resources/B.jpg")]
colors = []
pic = 0
for img in images:
    img = cv2.resize(img, (600, 750))
    
    blues = pro_blue(img)
    reds = pro_red(img)
    greens = pro_green(img)
    oranges = pro_orange(img)
    whites = pro_white(img)
    
    squares = []
    cont_b = get_contours(blues)
    cont_r = get_contours(reds)
    cont_g = get_contours(greens)
    cont_o = get_contours(oranges)
    cont_w = get_contours(whites)
    squares.append(cont_b)
    squares.append(cont_r)
    squares.append(cont_g)
    squares.append(cont_o)
    squares.append(cont_w)
    squares = resquare(squares)
    
    for i in range(9):
        squares[i] = reorder_points(squares[i])
    squares = reorder_squares(squares)
    
    colors.extend(get_colors(squares, cv2.cvtColor(img, cv2.COLOR_BGR2HSV)))
    pic += 1
init_state=""
init_state = init_state.join(colors)
print(init_state)

