#             |************|
#             |*U1**U2**U3*|
#             |************|
#             |*U4**U5**U6*|
#             |************|
#             |*U7**U8**U9*|
#             |************|
# ************|************|************|************
# *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*
# ************|************|************|************
# *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*
# ************|************|************|************
# *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*
# ************|************|************|************
#             |************|
#             |*D1**D2**D3*|
#             |************|
#             |*D4**D5**D6*|
#             |************|
#             |*D7**D8**D9*|
#             |************|

# U -> Blanco
# R -> Rojo
# F -> Verde
# D -> Amarillo
# L -> Naranja
# B -> Azul

# documentation
# http://pyopengl.sourceforge.net/documentation/manual-3.0/index.html
# https://www.pygame.org/docs/
# https://pypi.org/project/kociemba/

import random
import cv2
import numpy as np
import pygame
import kociemba
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

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
        # if(index != 4):
        for i in range(10):
            avg_h += hsv_img[random.randrange(min_1, max_1)][random.randrange(min_0, max_0)][0]
            avg_s += hsv_img[random.randrange(min_1, max_1)][random.randrange(min_0, max_0)][1]
        avg_h = avg_h / 10
        avg_s = avg_s / 10
        '''else:
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
        '''
        if avg_s < 85:
            list_out.append('F')
        elif avg_h < 20:
            list_out.append('L')
        elif avg_h < 45:
            list_out.append('B')
        elif avg_h < 90:
            list_out.append('D')
        elif avg_h < 153:
            list_out.append('U')
            print(avg_h)
        else:
            list_out.append('R')
        index += 1
    return list_out


images = [cv2.imread("Resources/U.jpg"), cv2.imread("Resources/R.jpg"), cv2.imread("Resources/F.jpg"), cv2.imread("Resources/D.jpg"), cv2.imread("Resources/L.jpg"), cv2.imread("Resources/B.jpg")]
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

vertices = (
    # U1
    (-6, 10, 0),
    (-4, 10, 0),
    (-6, 8, 0),
    (-4, 8, 0),
    # U2
    (-4, 10, 0),
    (-2, 10, 0),
    (-4, 8, 0),
    (-2, 8, 0),
    # U3
    (-2, 10, 0),
    (0, 10, 0),
    (-2, 8, 0),
    (0, 8, 0),
    # U4
    (-6, 8, 0),
    (-4, 8, 0),
    (-6, 6, 0),
    (-4, 6, 0),
    # U5
    (-4, 8, 0),
    (-2, 8, 0),
    (-4, 6, 0),
    (-2, 6, 0),
    # U6
    (-2, 8, 0),
    (0, 8, 0),
    (-2, 6, 0),
    (0, 6, 0),
    # U7
    (-6, 6, 0),
    (-4, 6, 0),
    (-6, 4, 0),
    (-4, 4, 0),
    # U8
    (-4, 6, 0),
    (-2, 6, 0),
    (-4, 4, 0),
    (-2, 4, 0),
    # U9
    (-2, 6, 0),
    (0, 6, 0),
    (-2, 4, 0),
    (0, 4, 0),
    # R1
    (0, 4, 0),
    (2, 4, 0),
    (0, 2, 0),
    (2, 2, 0),
    # R2
    (2, 4, 0),
    (4, 4, 0),
    (2, 2, 0),
    (4, 2, 0),
    # R3
    (4, 4, 0),
    (6, 4, 0),
    (4, 2, 0),
    (6, 2, 0),
    # R4
    (0, 2, 0),
    (2, 2, 0),
    (0, 0, 0),
    (2, 0, 0),
    # R5
    (2, 2, 0),
    (4, 2, 0),
    (2, 0, 0),
    (4, 0, 0),
    # R6
    (4, 2, 0),
    (6, 2, 0),
    (4, 0, 0),
    (6, 0, 0),
    # R7
    (0, 0, 0),
    (2, 0, 0),
    (0, -2, 0),
    (2, -2, 0),
    # R8
    (2, 0, 0),
    (4, 0, 0),
    (2, -2, 0),
    (4, -2, 0),
    # R9
    (4, 0, 0),
    (6, 0, 0),
    (4, -2, 0),
    (6, -2, 0),
    # F1
    (-6, 4, 0),
    (-4, 4, 0),
    (-6, 2, 0),
    (-4, 2, 0),
    # F2
    (-4, 4, 0),
    (-2, 4, 0),
    (-4, 2, 0),
    (-2, 2, 0),
    # F3
    (-2, 4, 0),
    (0, 4, 0),
    (-2, 2, 0),
    (0, 2, 0),
    # F4
    (-6, 2, 0),
    (-4, 2, 0),
    (-6, 0, 0),
    (-4, 0, 0),
    # F5
    (-4, 2, 0),
    (-2, 2, 0),
    (-4, 0, 0),
    (-2, 0, 0),
    # F6
    (-2, 2, 0),
    (0, 2, 0),
    (-2, 0, 0),
    (0, 0, 0),
    # F7
    (-6, 0, 0),
    (-4, 0, 0),
    (-6, -2, 0),
    (-4, -2, 0),
    # F8
    (-4, 0, 0),
    (-2, 0, 0),
    (-4, -2, 0),
    (-2, -2, 0),
    # F9
    (-2, 0, 0),
    (0, 0, 0),
    (-2, -2, 0),
    (0, -2, 0),
    # D1
    (-6, -2, 0),
    (-4, -2, 0),
    (-6, -4, 0),
    (-4, -4, 0),
    # D2
    (-4, -2, 0),
    (-2, -2, 0),
    (-4, -4, 0),
    (-2, -4, 0),
    # D3
    (-2, -2, 0),
    (0, -2, 0),
    (-2, -4, 0),
    (0, -4, 0),
    # D4
    (-6, -4, 0),
    (-4, -4, 0),
    (-6, -6, 0),
    (-4, -6, 0),
    # D5
    (-4, -4, 0),
    (-2, -4, 0),
    (-4, -6, 0),
    (-2, -6, 0),
    # D6
    (-2, -4, 0),
    (0, -4, 0),
    (-2, -6, 0),
    (0, -6, 0),
    # D7
    (-6, -6, 0),
    (-4, -6, 0),
    (-6, -8, 0),
    (-4, -8, 0),
    # D8
    (-4, -6, 0),
    (-2, -6, 0),
    (-4, -8, 0),
    (-2, -8, 0),
    # D9
    (-2, -6, 0),
    (0, -6, 0),
    (-2, -8, 0),
    (0, -8, 0),
    # L1
    (-12, 4, 0),
    (-10, 4, 0),
    (-12, 2, 0),
    (-10, 2, 0),
    # L2
    (-10, 4, 0),
    (-8, 4, 0),
    (-10, 2, 0),
    (-8, 2, 0),
    # L3
    (-8, 4, 0),
    (-6, 4, 0),
    (-8, 2, 0),
    (-6, 2, 0),
    # L4
    (-12, 2, 0),
    (-10, 2, 0),
    (-12, 0, 0),
    (-10, 0, 0),
    # L5
    (-10, 2, 0),
    (-8, 2, 0),
    (-10, 0, 0),
    (-8, 0, 0),
    # L6
    (-8, 2, 0),
    (-6, 2, 0),
    (-8, 0, 0),
    (-6, 0, 0),
    # L7
    (-12, 0, 0),
    (-10, 0, 0),
    (-12, -2, 0),
    (-10, -2, 0),
    # L8
    (-10, 0, 0),
    (-8, 0, 0),
    (-10, -2, 0),
    (-8, -2, 0),
    # L9
    (-8, 0, 0),
    (-6, 0, 0),
    (-8, -2, 0),
    (-6, -2, 0),
    # B1
    (8, 4, 0),
    (6, 4, 0),
    (8, 2, 0),
    (6, 2, 0),
    # B2
    (10, 4, 0),
    (8, 4, 0),
    (10, 2, 0),
    (8, 2, 0),
    # B3
    (12, 4, 0),
    (10, 4, 0),
    (12, 2, 0),
    (10, 2, 0),
    # B4
    (6, 2, 0),
    (8, 2, 0),
    (6, 0, 0),
    (8, 0, 0),
    # B5
    (8, 2, 0),
    (10, 2, 0),
    (8, 0, 0),
    (10, 0, 0),
    # B6
    (10, 2, 0),
    (12, 2, 0),
    (10, 0, 0),
    (12, 0, 0),
    # B7
    (6, 0, 0),
    (8, 0, 0),
    (6, -2, 0),
    (8, -2, 0),
    # B8
    (8, 0, 0),
    (10, 0, 0),
    (8, -2, 0),
    (10, -2, 0),
    # B9
    (10, 0, 0),
    (12, 0, 0),
    (10, -2, 0),
    (12, -2, 0)
)

edges = (
    # U1
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    # U2
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    # U3
    (8, 9),
    (9, 11),
    (11, 10),
    (10, 8),
    # U4
    (12, 13),
    (13, 15),
    (15, 14),
    (14, 12),
    # U5
    (16, 17),
    (17, 19),
    (19, 18),
    (18, 16),
    # U6
    (20, 21),
    (21, 23),
    (23, 22),
    (22, 20),
    # U7
    (24, 25),
    (25, 27),
    (27, 26),
    (26, 24),
    # U8
    (28, 29),
    (29, 31),
    (31, 30),
    (30, 28),
    # U9
    (32, 33),
    (33, 35),
    (35, 34),
    (34, 32),
    # R1
    (36, 37),
    (37, 39),
    (39, 38),
    (38, 36),
    # R2
    (40, 41),
    (41, 43),
    (43, 42),
    (42, 40),
    # R3
    (44, 45),
    (45, 47),
    (47, 46),
    (46, 44),
    # R4
    (48, 49),
    (49, 51),
    (51, 50),
    (50, 48),
    # R5
    (52, 53),
    (53, 55),
    (55, 54),
    (54, 52),
    # R6
    (56, 57),
    (57, 59),
    (59, 58),
    (58, 56),
    # R7
    (60, 61),
    (61, 63),
    (63, 62),
    (62, 60),
    # R8
    (64, 65),
    (65, 67),
    (67, 66),
    (66, 64),
    # R9
    (68, 69),
    (69, 71),
    (71, 70),
    (70, 68),
    # F1
    (72, 73),
    (73, 75),
    (75, 74),
    (74, 72),
    # F2
    (76, 77),
    (77, 79),
    (79, 78),
    (78, 76),
    # F3
    (80, 81),
    (81, 83),
    (83, 82),
    (82, 80),
    # F4
    (84, 85),
    (85, 87),
    (87, 86),
    (86, 84),
    # F5
    (88, 89),
    (89, 91),
    (91, 90),
    (90, 88),
    # F6
    (92, 93),
    (93, 95),
    (95, 94),
    (94, 92),
    # F7
    (96, 97),
    (97, 99),
    (99, 98),
    (98, 96),
    # F8
    (100, 101),
    (101, 103),
    (103, 102),
    (102, 100),
    # F9
    (104, 105),
    (105, 107),
    (107, 106),
    (106, 104),
    # D1
    (108, 109),
    (109, 111),
    (111, 110),
    (110, 108),
    # D2
    (112, 113),
    (113, 115),
    (115, 114),
    (114, 112),
    # D3
    (116, 117),
    (117, 119),
    (119, 118),
    (118, 116),
    # D4
    (120, 121),
    (121, 123),
    (123, 122),
    (122, 120),
    # D5
    (124, 125),
    (125, 127),
    (127, 126),
    (126, 124),
    # D6
    (128, 129),
    (129, 131),
    (131, 130),
    (130, 128),
    # D7
    (132, 133),
    (133, 135),
    (135, 134),
    (134, 132),
    # D8
    (136, 137),
    (137, 139),
    (139, 138),
    (138, 136),
    # D9
    (140, 141),
    (141, 143),
    (143, 142),
    (142, 140),
    # L1
    (144, 145),
    (145, 147),
    (147, 146),
    (146, 144),
    # L2
    (148, 149),
    (149, 151),
    (151, 150),
    (150, 148),
    # L3
    (152, 153),
    (153, 155),
    (155, 154),
    (154, 152),
    # L4
    (156, 157),
    (157, 159),
    (159, 158),
    (158, 156),
    # L5
    (160, 161),
    (161, 163),
    (163, 162),
    (162, 160),
    # L6
    (164, 165),
    (165, 167),
    (167, 166),
    (166, 164),
    # L7
    (168, 169),
    (169, 171),
    (171, 170),
    (170, 168),
    # L8
    (172, 173),
    (173, 175),
    (175, 174),
    (174, 172),
    # L9
    (176, 177),
    (177, 179),
    (179, 178),
    (178, 176),
    # B1
    (180, 181),
    (181, 183),
    (183, 182),
    (182, 180),
    # B2
    (184, 185),
    (185, 187),
    (187, 186),
    (186, 184),
    # B3
    (188, 189),
    (189, 191),
    (191, 190),
    (190, 188),
    # B4
    (192, 193),
    (193, 195),
    (195, 194),
    (194, 192),
    # B5
    (196, 197),
    (197, 199),
    (199, 198),
    (198, 196),
    # B6
    (200, 201),
    (201, 203),
    (203, 202),
    (202, 200),
    # B7
    (204, 205),
    (205, 207),
    (207, 206),
    (206, 204),
    # B8
    (208, 209),
    (209, 211),
    (211, 210),
    (210, 208),
    # B9
    (212, 213),
    (213, 215),
    (215, 214),
    (214, 212)
)

surfaces = (
    # U
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (8, 9, 11, 10),
    (12, 13, 15, 14),
    (16, 17, 19, 18),
    (20, 21, 23, 22),
    (24, 25, 27, 26),
    (28, 29, 31, 30),
    (32, 33, 35, 34),
    # R
    (36, 37, 39, 38),
    (40, 41, 43, 42),
    (44, 45, 47, 46),
    (48, 49, 51, 50),
    (52, 53, 55, 54),
    (56, 57, 59, 58),
    (60, 61, 63, 62),
    (64, 65, 67, 66),
    (68, 69, 71, 70),
    # F
    (72, 73, 75, 74),
    (76, 77, 79, 78),
    (80, 81, 83, 82),
    (84, 85, 87, 86),
    (88, 89, 91, 90),
    (92, 93, 95, 94),
    (96, 97, 99, 98),
    (100, 101, 103, 102),
    (104, 105, 107, 106),
    # D
    (108, 109, 111, 110),
    (112, 113, 115, 114),
    (116, 117, 119, 118),
    (120, 121, 123, 122),
    (124, 125, 127, 126),
    (128, 129, 131, 130),
    (132, 133, 135, 134),
    (136, 137, 139, 138),
    (140, 141, 143, 142),
    # L
    (144, 145, 147, 146),
    (148, 149, 151, 150),
    (152, 153, 155, 154),
    (156, 157, 159, 158),
    (160, 161, 163, 162),
    (164, 165, 167, 166),
    (168, 169, 171, 170),
    (172, 173, 175, 174),
    (176, 177, 179, 178),
    # B
    (180, 181, 183, 182),
    (184, 185, 187, 186),
    (188, 189, 191, 190),
    (192, 193, 195, 194),
    (196, 197, 199, 198),
    (200, 201, 203, 202),
    (204, 205, 207, 206),
    (208, 209, 211, 210),
    (212, 213, 215, 214)
)


def Cube(estado):
    i = 0
    j = 0
    glBegin(GL_QUADS)
    for surface in surfaces:
        for vertex in surface:
            # F -> Blanco
            if estado[i] == 'F':
                glColor3fv((255, 255, 255))
                glVertex3fv(vertices[vertex])
            # R -> Rojo
            if estado[i] == 'R':
                glColor3fv((255, 0, 0))
                glVertex3fv(vertices[vertex])
            # D -> Verde
            if estado[i] == 'D':
                glColor3fv((0, 128, 0))
                glVertex3fv(vertices[vertex])
            # B -> Amarillo
            if estado[i] == 'B':
                glColor3fv((255, 255, 0))
                glVertex3fv(vertices[vertex])
            # L -> Purpura (Naranja)
            if estado[i] == 'L':
                glColor3fv((128, 0, 128))
                glVertex3fv(vertices[vertex])
            # U -> Azul
            if estado[i] == 'U':
                glColor3fv((0, 0, 255))
                glVertex3fv(vertices[vertex])
            j = j+1
            if j == 4:
                i = i+1
                j = 0

    glEnd()

    # glBegin: delimit the vertices of a primitive or a group of like primitives
    x = 1
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            # specify a vertex
            glColor3f(0, 0, 0)
            glVertex3fv(vertices[vertex])
    glEnd()


def insert(cadena, pos, char):
    string = list(cadena)
    string[pos] = char
    new_string = "".join(string)
    return new_string


def partition(cadena):
    part = cadena.partition(' ')
    print(part)
    return part[0], part[2]


def main(s, p):
    estado = s
    pasos = p
    paso_actual = ""
    pygame.init()
    display = (900, 600)
    # Initialize a window or screen for display
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    # set up a perspective projection matrix
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    # multiply the current matrix by a translation matrix
    glTranslatef(0.0, 0.0, -30)
    # multiply the current matrix by a rotation matrix
    glRotatef(0, 0, 0, 0)
    while True:
        temp = ['', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '']
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                paso_actual, pasos = partition(pasos)
                if event.key == pygame.K_RIGHT:
                    # Front Clockwise
                    if paso_actual == 'F':
                        temp[0], temp[1], temp[2] = estado[6], estado[7], estado[8]
                        temp[3], temp[4], temp[5] = estado[9], estado[12], estado[15]
                        temp[6], temp[7], temp[8] = estado[29], estado[28], estado[27]
                        temp[9], temp[10], temp[11] = estado[44], estado[41], estado[38]
                        temp[12], temp[13], temp[14] = estado[18], estado[19], estado[20]
                        temp[15], temp[16], temp[17] = estado[23], estado[26], estado[25]
                        temp[18], temp[19] = estado[24], estado[21]

                        estado = insert(estado, 9, temp[0])
                        estado = insert(estado, 12, temp[1])
                        estado = insert(estado, 15, temp[2])
                        estado = insert(estado, 29, temp[3])
                        estado = insert(estado, 28, temp[4])
                        estado = insert(estado, 27, temp[5])
                        estado = insert(estado, 44, temp[6])
                        estado = insert(estado, 41, temp[7])
                        estado = insert(estado, 38, temp[8])
                        estado = insert(estado, 6, temp[9])
                        estado = insert(estado, 7, temp[10])
                        estado = insert(estado, 8, temp[11])
                        estado = insert(estado, 20, temp[12])
                        estado = insert(estado, 23, temp[13])
                        estado = insert(estado, 26, temp[14])
                        estado = insert(estado, 25, temp[15])
                        estado = insert(estado, 24, temp[16])
                        estado = insert(estado, 21, temp[17])
                        estado = insert(estado, 18, temp[18])
                        estado = insert(estado, 19, temp[19])
                    # Right Clockwise
                    if paso_actual == 'R':
                        temp[0], temp[1], temp[2] = estado[8], estado[5], estado[2]
                        temp[3], temp[4], temp[5] = estado[45], estado[48], estado[51]
                        temp[6], temp[7], temp[8] = estado[35], estado[32], estado[29]
                        temp[9], temp[10], temp[11] = estado[26], estado[23], estado[20]
                        temp[12], temp[13], temp[14] = estado[9], estado[10], estado[11]
                        temp[15], temp[16], temp[17] = estado[14], estado[17], estado[16]
                        temp[18], temp[19] = estado[15], estado[12]

                        estado = insert(estado, 45, temp[0])
                        estado = insert(estado, 48, temp[1])
                        estado = insert(estado, 51, temp[2])
                        estado = insert(estado, 35, temp[3])
                        estado = insert(estado, 32, temp[4])
                        estado = insert(estado, 29, temp[5])
                        estado = insert(estado, 26, temp[6])
                        estado = insert(estado, 23, temp[7])
                        estado = insert(estado, 20, temp[8])
                        estado = insert(estado, 8, temp[9])
                        estado = insert(estado, 5, temp[10])
                        estado = insert(estado, 2, temp[11])
                        estado = insert(estado, 11, temp[12])
                        estado = insert(estado, 14, temp[13])
                        estado = insert(estado, 17, temp[14])
                        estado = insert(estado, 16, temp[15])
                        estado = insert(estado, 15, temp[16])
                        estado = insert(estado, 12, temp[17])
                        estado = insert(estado, 9, temp[18])
                        estado = insert(estado, 10, temp[19])
                    # Up Clockwise
                    if paso_actual == 'U':
                        temp[0], temp[1], temp[2] = estado[38], estado[37], estado[36]
                        temp[3], temp[4], temp[5] = estado[47], estado[46], estado[45]
                        temp[6], temp[7], temp[8] = estado[11], estado[10], estado[9]
                        temp[9], temp[10], temp[11] = estado[20], estado[19], estado[18]
                        temp[12], temp[13], temp[14] = estado[0], estado[1], estado[2]
                        temp[15], temp[16], temp[17] = estado[5], estado[8], estado[7]
                        temp[18], temp[19] = estado[6], estado[3]

                        estado = insert(estado, 47, temp[0])
                        estado = insert(estado, 46, temp[1])
                        estado = insert(estado, 45, temp[2])
                        estado = insert(estado, 11, temp[3])
                        estado = insert(estado, 10, temp[4])
                        estado = insert(estado, 9, temp[5])
                        estado = insert(estado, 20, temp[6])
                        estado = insert(estado, 19, temp[7])
                        estado = insert(estado, 18, temp[8])
                        estado = insert(estado, 38, temp[9])
                        estado = insert(estado, 37, temp[10])
                        estado = insert(estado, 36, temp[11])
                        estado = insert(estado, 2, temp[12])
                        estado = insert(estado, 5, temp[13])
                        estado = insert(estado, 8, temp[14])
                        estado = insert(estado, 7, temp[15])
                        estado = insert(estado, 6, temp[16])
                        estado = insert(estado, 3, temp[17])
                        estado = insert(estado, 0, temp[18])
                        estado = insert(estado, 1, temp[19])
                    # Back Clockwise
                    if paso_actual == 'B':
                        temp[0], temp[1], temp[2] = estado[2], estado[1], estado[0]
                        temp[3], temp[4], temp[5] = estado[36], estado[39], estado[42]
                        temp[6], temp[7], temp[8] = estado[33], estado[34], estado[35]
                        temp[9], temp[10], temp[11] = estado[17], estado[14], estado[11]
                        temp[12], temp[13], temp[14] = estado[45], estado[46], estado[47]
                        temp[15], temp[16], temp[17] = estado[50], estado[53], estado[52]
                        temp[18], temp[19] = estado[51], estado[48]

                        estado = insert(estado, 36, temp[0])
                        estado = insert(estado, 39, temp[1])
                        estado = insert(estado, 42, temp[2])
                        estado = insert(estado, 33, temp[3])
                        estado = insert(estado, 34, temp[4])
                        estado = insert(estado, 35, temp[5])
                        estado = insert(estado, 17, temp[6])
                        estado = insert(estado, 14, temp[7])
                        estado = insert(estado, 11, temp[8])
                        estado = insert(estado, 2, temp[9])
                        estado = insert(estado, 1, temp[10])
                        estado = insert(estado, 0, temp[11])
                        estado = insert(estado, 47, temp[12])
                        estado = insert(estado, 50, temp[13])
                        estado = insert(estado, 53, temp[14])
                        estado = insert(estado, 52, temp[15])
                        estado = insert(estado, 51, temp[16])
                        estado = insert(estado, 48, temp[17])
                        estado = insert(estado, 45, temp[18])
                        estado = insert(estado, 46, temp[19])
                    # Left Clockwise
                    if paso_actual == 'L':
                        temp[0], temp[1], temp[2] = estado[0], estado[3], estado[6]
                        temp[3], temp[4], temp[5] = estado[18], estado[21], estado[24]
                        temp[6], temp[7], temp[8] = estado[27], estado[30], estado[33]
                        temp[9], temp[10], temp[11] = estado[53], estado[50], estado[47]
                        temp[12], temp[13], temp[14] = estado[36], estado[37], estado[38]
                        temp[15], temp[16], temp[17] = estado[41], estado[44], estado[43]
                        temp[18], temp[19] = estado[42], estado[39]

                        estado = insert(estado, 18, temp[0])
                        estado = insert(estado, 21, temp[1])
                        estado = insert(estado, 24, temp[2])
                        estado = insert(estado, 27, temp[3])
                        estado = insert(estado, 30, temp[4])
                        estado = insert(estado, 33, temp[5])
                        estado = insert(estado, 53, temp[6])
                        estado = insert(estado, 50, temp[7])
                        estado = insert(estado, 47, temp[8])
                        estado = insert(estado, 0, temp[9])
                        estado = insert(estado, 3, temp[10])
                        estado = insert(estado, 6, temp[11])
                        estado = insert(estado, 38, temp[12])
                        estado = insert(estado, 41, temp[13])
                        estado = insert(estado, 44, temp[14])
                        estado = insert(estado, 43, temp[15])
                        estado = insert(estado, 42, temp[16])
                        estado = insert(estado, 39, temp[17])
                        estado = insert(estado, 36, temp[18])
                        estado = insert(estado, 37, temp[19])
                    # Down Clockwise
                    if paso_actual == 'D':
                        temp[0], temp[1], temp[2] = estado[42], estado[43], estado[44]
                        temp[3], temp[4], temp[5] = estado[24], estado[25], estado[26]
                        temp[6], temp[7], temp[8] = estado[15], estado[16], estado[17]
                        temp[9], temp[10], temp[11] = estado[51], estado[52], estado[53]
                        temp[12], temp[13], temp[14] = estado[27], estado[28], estado[29]
                        temp[15], temp[16], temp[17] = estado[32], estado[35], estado[34]
                        temp[18], temp[19] = estado[33], estado[30]

                        estado = insert(estado, 24, temp[0])
                        estado = insert(estado, 25, temp[1])
                        estado = insert(estado, 26, temp[2])
                        estado = insert(estado, 15, temp[3])
                        estado = insert(estado, 16, temp[4])
                        estado = insert(estado, 17, temp[5])
                        estado = insert(estado, 51, temp[6])
                        estado = insert(estado, 52, temp[7])
                        estado = insert(estado, 53, temp[8])
                        estado = insert(estado, 42, temp[9])
                        estado = insert(estado, 43, temp[10])
                        estado = insert(estado, 44, temp[11])
                        estado = insert(estado, 29, temp[12])
                        estado = insert(estado, 32, temp[13])
                        estado = insert(estado, 35, temp[14])
                        estado = insert(estado, 34, temp[15])
                        estado = insert(estado, 33, temp[16])
                        estado = insert(estado, 30, temp[17])
                        estado = insert(estado, 27, temp[18])
                        estado = insert(estado, 28, temp[19])
                    # Front Counterclockwise
                    if paso_actual == 'F\'':
                        temp[0], temp[1], temp[2] = estado[8], estado[7], estado[6]
                        temp[3], temp[4], temp[5] = estado[38], estado[41], estado[44]
                        temp[6], temp[7], temp[8] = estado[27], estado[28], estado[29]
                        temp[9], temp[10], temp[11] = estado[15], estado[12], estado[9]
                        temp[12], temp[13], temp[14] = estado[20], estado[19], estado[18]
                        temp[15], temp[16], temp[17] = estado[21], estado[24], estado[25]
                        temp[18], temp[19] = estado[26], estado[23]

                        estado = insert(estado, 38, temp[0])
                        estado = insert(estado, 41, temp[1])
                        estado = insert(estado, 44, temp[2])
                        estado = insert(estado, 27, temp[3])
                        estado = insert(estado, 28, temp[4])
                        estado = insert(estado, 29, temp[5])
                        estado = insert(estado, 15, temp[6])
                        estado = insert(estado, 12, temp[7])
                        estado = insert(estado, 9, temp[8])
                        estado = insert(estado, 8, temp[9])
                        estado = insert(estado, 7, temp[10])
                        estado = insert(estado, 6, temp[11])
                        estado = insert(estado, 18, temp[12])
                        estado = insert(estado, 21, temp[13])
                        estado = insert(estado, 24, temp[14])
                        estado = insert(estado, 25, temp[15])
                        estado = insert(estado, 26, temp[16])
                        estado = insert(estado, 23, temp[17])
                        estado = insert(estado, 20, temp[18])
                        estado = insert(estado, 19, temp[19])
                    # Right Counterclockwise
                    if paso_actual == 'R\'':
                        temp[0], temp[1], temp[2] = estado[2], estado[5], estado[8]
                        temp[3], temp[4], temp[5] = estado[20], estado[23], estado[26]
                        temp[6], temp[7], temp[8] = estado[29], estado[32], estado[35]
                        temp[9], temp[10], temp[11] = estado[51], estado[48], estado[45]
                        temp[12], temp[13], temp[14] = estado[11], estado[10], estado[9]
                        temp[15], temp[16], temp[17] = estado[12], estado[15], estado[16]
                        temp[18], temp[19] = estado[17], estado[14]

                        estado = insert(estado, 20, temp[0])
                        estado = insert(estado, 23, temp[1])
                        estado = insert(estado, 26, temp[2])
                        estado = insert(estado, 29, temp[3])
                        estado = insert(estado, 32, temp[4])
                        estado = insert(estado, 35, temp[5])
                        estado = insert(estado, 51, temp[6])
                        estado = insert(estado, 48, temp[7])
                        estado = insert(estado, 45, temp[8])
                        estado = insert(estado, 2, temp[9])
                        estado = insert(estado, 5, temp[10])
                        estado = insert(estado, 8, temp[11])
                        estado = insert(estado, 9, temp[12])
                        estado = insert(estado, 12, temp[13])
                        estado = insert(estado, 15, temp[14])
                        estado = insert(estado, 16, temp[15])
                        estado = insert(estado, 17, temp[16])
                        estado = insert(estado, 14, temp[17])
                        estado = insert(estado, 11, temp[18])
                        estado = insert(estado, 10, temp[19])
                    # Up Counterclockwise
                    if paso_actual == 'U\'':
                        temp[0], temp[1], temp[2] = estado[36], estado[37], estado[38]
                        temp[3], temp[4], temp[5] = estado[18], estado[19], estado[20]
                        temp[6], temp[7], temp[8] = estado[9], estado[10], estado[11]
                        temp[9], temp[10], temp[11] = estado[45], estado[46], estado[47]
                        temp[12], temp[13], temp[14] = estado[2], estado[1], estado[0]
                        temp[15], temp[16], temp[17] = estado[3], estado[6], estado[7]
                        temp[18], temp[19] = estado[8], estado[5]

                        estado = insert(estado, 18, temp[0])
                        estado = insert(estado, 19, temp[1])
                        estado = insert(estado, 20, temp[2])
                        estado = insert(estado, 9, temp[3])
                        estado = insert(estado, 10, temp[4])
                        estado = insert(estado, 11, temp[5])
                        estado = insert(estado, 45, temp[6])
                        estado = insert(estado, 46, temp[7])
                        estado = insert(estado, 47, temp[8])
                        estado = insert(estado, 36, temp[9])
                        estado = insert(estado, 37, temp[10])
                        estado = insert(estado, 38, temp[11])
                        estado = insert(estado, 0, temp[12])
                        estado = insert(estado, 3, temp[13])
                        estado = insert(estado, 6, temp[14])
                        estado = insert(estado, 7, temp[15])
                        estado = insert(estado, 8, temp[16])
                        estado = insert(estado, 5, temp[17])
                        estado = insert(estado, 2, temp[18])
                        estado = insert(estado, 1, temp[19])
                    # Back Counterclockwise
                    if paso_actual == 'B\'':
                        temp[0], temp[1], temp[2] = estado[0], estado[1], estado[2]
                        temp[3], temp[4], temp[5] = estado[11], estado[14], estado[17]
                        temp[6], temp[7], temp[8] = estado[35], estado[34], estado[33]
                        temp[9], temp[10], temp[11] = estado[42], estado[39], estado[36]
                        temp[12], temp[13], temp[14] = estado[47], estado[46], estado[45]
                        temp[15], temp[16], temp[17] = estado[48], estado[51], estado[52]
                        temp[18], temp[19] = estado[53], estado[50]

                        estado = insert(estado, 11, temp[0])
                        estado = insert(estado, 14, temp[1])
                        estado = insert(estado, 17, temp[2])
                        estado = insert(estado, 35, temp[3])
                        estado = insert(estado, 34, temp[4])
                        estado = insert(estado, 33, temp[5])
                        estado = insert(estado, 42, temp[6])
                        estado = insert(estado, 39, temp[7])
                        estado = insert(estado, 36, temp[8])
                        estado = insert(estado, 0, temp[9])
                        estado = insert(estado, 1, temp[10])
                        estado = insert(estado, 2, temp[11])
                        estado = insert(estado, 45, temp[12])
                        estado = insert(estado, 48, temp[13])
                        estado = insert(estado, 51, temp[14])
                        estado = insert(estado, 52, temp[15])
                        estado = insert(estado, 53, temp[16])
                        estado = insert(estado, 50, temp[17])
                        estado = insert(estado, 47, temp[18])
                        estado = insert(estado, 46, temp[19])
                    # Left Counterclockwise
                    if paso_actual == 'L\'':
                        temp[0], temp[1], temp[2] = estado[47], estado[50], estado[53]
                        temp[3], temp[4], temp[5] = estado[33], estado[30], estado[27]
                        temp[6], temp[7], temp[8] = estado[24], estado[21], estado[18]
                        temp[9], temp[10], temp[11] = estado[6], estado[3], estado[0]
                        temp[12], temp[13], temp[14] = estado[38], estado[37], estado[36]
                        temp[15], temp[16], temp[17] = estado[39], estado[42], estado[43]
                        temp[18], temp[19] = estado[44], estado[41]

                        estado = insert(estado, 33, temp[0])
                        estado = insert(estado, 30, temp[1])
                        estado = insert(estado, 27, temp[2])
                        estado = insert(estado, 24, temp[3])
                        estado = insert(estado, 21, temp[4])
                        estado = insert(estado, 18, temp[5])
                        estado = insert(estado, 6, temp[6])
                        estado = insert(estado, 3, temp[7])
                        estado = insert(estado, 0, temp[8])
                        estado = insert(estado, 47, temp[9])
                        estado = insert(estado, 50, temp[10])
                        estado = insert(estado, 53, temp[11])
                        estado = insert(estado, 36, temp[12])
                        estado = insert(estado, 39, temp[13])
                        estado = insert(estado, 42, temp[14])
                        estado = insert(estado, 43, temp[15])
                        estado = insert(estado, 44, temp[16])
                        estado = insert(estado, 41, temp[17])
                        estado = insert(estado, 38, temp[18])
                        estado = insert(estado, 37, temp[19])
                    # Left Counterclockwise
                    if paso_actual == 'D\'':
                        temp[0], temp[1], temp[2] = estado[53], estado[52], estado[51]
                        temp[3], temp[4], temp[5] = estado[17], estado[16], estado[15]
                        temp[6], temp[7], temp[8] = estado[26], estado[25], estado[24]
                        temp[9], temp[10], temp[11] = estado[44], estado[43], estado[42]
                        temp[12], temp[13], temp[14] = estado[29], estado[28], estado[27]
                        temp[15], temp[16], temp[17] = estado[30], estado[33], estado[34]
                        temp[18], temp[19] = estado[35], estado[32]

                        estado = insert(estado, 17, temp[0])
                        estado = insert(estado, 16, temp[1])
                        estado = insert(estado, 15, temp[2])
                        estado = insert(estado, 26, temp[3])
                        estado = insert(estado, 25, temp[4])
                        estado = insert(estado, 24, temp[5])
                        estado = insert(estado, 44, temp[6])
                        estado = insert(estado, 43, temp[7])
                        estado = insert(estado, 42, temp[8])
                        estado = insert(estado, 53, temp[9])
                        estado = insert(estado, 52, temp[10])
                        estado = insert(estado, 51, temp[11])
                        estado = insert(estado, 27, temp[12])
                        estado = insert(estado, 30, temp[13])
                        estado = insert(estado, 33, temp[14])
                        estado = insert(estado, 34, temp[15])
                        estado = insert(estado, 35, temp[16])
                        estado = insert(estado, 32, temp[17])
                        estado = insert(estado, 29, temp[18])
                        estado = insert(estado, 28, temp[19])
                    # Front 180
                    if paso_actual == 'F2':
                        temp[0], temp[1], temp[2] = estado[6], estado[7], estado[8]
                        temp[3], temp[4], temp[5] = estado[9], estado[12], estado[15]
                        temp[6], temp[7], temp[8] = estado[29], estado[28], estado[27]
                        temp[9], temp[10], temp[11] = estado[44], estado[41], estado[38]
                        temp[12], temp[13], temp[14] = estado[18], estado[19], estado[20]
                        temp[15], temp[16], temp[17] = estado[23], estado[26], estado[25]
                        temp[18], temp[19] = estado[24], estado[21]

                        estado = insert(estado, 9, temp[0])
                        estado = insert(estado, 12, temp[1])
                        estado = insert(estado, 15, temp[2])
                        estado = insert(estado, 29, temp[3])
                        estado = insert(estado, 28, temp[4])
                        estado = insert(estado, 27, temp[5])
                        estado = insert(estado, 44, temp[6])
                        estado = insert(estado, 41, temp[7])
                        estado = insert(estado, 38, temp[8])
                        estado = insert(estado, 6, temp[9])
                        estado = insert(estado, 7, temp[10])
                        estado = insert(estado, 8, temp[11])
                        estado = insert(estado, 20, temp[12])
                        estado = insert(estado, 23, temp[13])
                        estado = insert(estado, 26, temp[14])
                        estado = insert(estado, 25, temp[15])
                        estado = insert(estado, 24, temp[16])
                        estado = insert(estado, 21, temp[17])
                        estado = insert(estado, 18, temp[18])
                        estado = insert(estado, 19, temp[19])

                        temp[0], temp[1], temp[2] = estado[6], estado[7], estado[8]
                        temp[3], temp[4], temp[5] = estado[9], estado[12], estado[15]
                        temp[6], temp[7], temp[8] = estado[29], estado[28], estado[27]
                        temp[9], temp[10], temp[11] = estado[44], estado[41], estado[38]
                        temp[12], temp[13], temp[14] = estado[18], estado[19], estado[20]
                        temp[15], temp[16], temp[17] = estado[23], estado[26], estado[25]
                        temp[18], temp[19] = estado[24], estado[21]

                        estado = insert(estado, 9, temp[0])
                        estado = insert(estado, 12, temp[1])
                        estado = insert(estado, 15, temp[2])
                        estado = insert(estado, 29, temp[3])
                        estado = insert(estado, 28, temp[4])
                        estado = insert(estado, 27, temp[5])
                        estado = insert(estado, 44, temp[6])
                        estado = insert(estado, 41, temp[7])
                        estado = insert(estado, 38, temp[8])
                        estado = insert(estado, 6, temp[9])
                        estado = insert(estado, 7, temp[10])
                        estado = insert(estado, 8, temp[11])
                        estado = insert(estado, 20, temp[12])
                        estado = insert(estado, 23, temp[13])
                        estado = insert(estado, 26, temp[14])
                        estado = insert(estado, 25, temp[15])
                        estado = insert(estado, 24, temp[16])
                        estado = insert(estado, 21, temp[17])
                        estado = insert(estado, 18, temp[18])
                        estado = insert(estado, 19, temp[19])
                    # Right 180
                    if paso_actual == 'R2':
                        temp[0], temp[1], temp[2] = estado[8], estado[5], estado[2]
                        temp[3], temp[4], temp[5] = estado[45], estado[48], estado[51]
                        temp[6], temp[7], temp[8] = estado[35], estado[32], estado[29]
                        temp[9], temp[10], temp[11] = estado[26], estado[23], estado[20]
                        temp[12], temp[13], temp[14] = estado[9], estado[10], estado[11]
                        temp[15], temp[16], temp[17] = estado[14], estado[17], estado[16]
                        temp[18], temp[19] = estado[15], estado[12]

                        estado = insert(estado, 45, temp[0])
                        estado = insert(estado, 48, temp[1])
                        estado = insert(estado, 51, temp[2])
                        estado = insert(estado, 35, temp[3])
                        estado = insert(estado, 32, temp[4])
                        estado = insert(estado, 29, temp[5])
                        estado = insert(estado, 26, temp[6])
                        estado = insert(estado, 23, temp[7])
                        estado = insert(estado, 20, temp[8])
                        estado = insert(estado, 8, temp[9])
                        estado = insert(estado, 5, temp[10])
                        estado = insert(estado, 2, temp[11])
                        estado = insert(estado, 11, temp[12])
                        estado = insert(estado, 14, temp[13])
                        estado = insert(estado, 17, temp[14])
                        estado = insert(estado, 16, temp[15])
                        estado = insert(estado, 15, temp[16])
                        estado = insert(estado, 12, temp[17])
                        estado = insert(estado, 9, temp[18])
                        estado = insert(estado, 10, temp[19])

                        temp[0], temp[1], temp[2] = estado[8], estado[5], estado[2]
                        temp[3], temp[4], temp[5] = estado[45], estado[48], estado[51]
                        temp[6], temp[7], temp[8] = estado[35], estado[32], estado[29]
                        temp[9], temp[10], temp[11] = estado[26], estado[23], estado[20]
                        temp[12], temp[13], temp[14] = estado[9], estado[10], estado[11]
                        temp[15], temp[16], temp[17] = estado[14], estado[17], estado[16]
                        temp[18], temp[19] = estado[15], estado[12]

                        estado = insert(estado, 45, temp[0])
                        estado = insert(estado, 48, temp[1])
                        estado = insert(estado, 51, temp[2])
                        estado = insert(estado, 35, temp[3])
                        estado = insert(estado, 32, temp[4])
                        estado = insert(estado, 29, temp[5])
                        estado = insert(estado, 26, temp[6])
                        estado = insert(estado, 23, temp[7])
                        estado = insert(estado, 20, temp[8])
                        estado = insert(estado, 8, temp[9])
                        estado = insert(estado, 5, temp[10])
                        estado = insert(estado, 2, temp[11])
                        estado = insert(estado, 11, temp[12])
                        estado = insert(estado, 14, temp[13])
                        estado = insert(estado, 17, temp[14])
                        estado = insert(estado, 16, temp[15])
                        estado = insert(estado, 15, temp[16])
                        estado = insert(estado, 12, temp[17])
                        estado = insert(estado, 9, temp[18])
                        estado = insert(estado, 10, temp[19])
                    # Up 180
                    if paso_actual == 'U2':
                        temp[0], temp[1], temp[2] = estado[38], estado[37], estado[36]
                        temp[3], temp[4], temp[5] = estado[47], estado[46], estado[45]
                        temp[6], temp[7], temp[8] = estado[11], estado[10], estado[9]
                        temp[9], temp[10], temp[11] = estado[20], estado[19], estado[18]
                        temp[12], temp[13], temp[14] = estado[0], estado[1], estado[2]
                        temp[15], temp[16], temp[17] = estado[5], estado[8], estado[7]
                        temp[18], temp[19] = estado[6], estado[3]

                        estado = insert(estado, 47, temp[0])
                        estado = insert(estado, 46, temp[1])
                        estado = insert(estado, 45, temp[2])
                        estado = insert(estado, 11, temp[3])
                        estado = insert(estado, 10, temp[4])
                        estado = insert(estado, 9, temp[5])
                        estado = insert(estado, 20, temp[6])
                        estado = insert(estado, 19, temp[7])
                        estado = insert(estado, 18, temp[8])
                        estado = insert(estado, 38, temp[9])
                        estado = insert(estado, 37, temp[10])
                        estado = insert(estado, 36, temp[11])
                        estado = insert(estado, 2, temp[12])
                        estado = insert(estado, 5, temp[13])
                        estado = insert(estado, 8, temp[14])
                        estado = insert(estado, 7, temp[15])
                        estado = insert(estado, 6, temp[16])
                        estado = insert(estado, 3, temp[17])
                        estado = insert(estado, 0, temp[18])
                        estado = insert(estado, 1, temp[19])

                        temp[0], temp[1], temp[2] = estado[38], estado[37], estado[36]
                        temp[3], temp[4], temp[5] = estado[47], estado[46], estado[45]
                        temp[6], temp[7], temp[8] = estado[11], estado[10], estado[9]
                        temp[9], temp[10], temp[11] = estado[20], estado[19], estado[18]
                        temp[12], temp[13], temp[14] = estado[0], estado[1], estado[2]
                        temp[15], temp[16], temp[17] = estado[5], estado[8], estado[7]
                        temp[18], temp[19] = estado[6], estado[3]

                        estado = insert(estado, 47, temp[0])
                        estado = insert(estado, 46, temp[1])
                        estado = insert(estado, 45, temp[2])
                        estado = insert(estado, 11, temp[3])
                        estado = insert(estado, 10, temp[4])
                        estado = insert(estado, 9, temp[5])
                        estado = insert(estado, 20, temp[6])
                        estado = insert(estado, 19, temp[7])
                        estado = insert(estado, 18, temp[8])
                        estado = insert(estado, 38, temp[9])
                        estado = insert(estado, 37, temp[10])
                        estado = insert(estado, 36, temp[11])
                        estado = insert(estado, 2, temp[12])
                        estado = insert(estado, 5, temp[13])
                        estado = insert(estado, 8, temp[14])
                        estado = insert(estado, 7, temp[15])
                        estado = insert(estado, 6, temp[16])
                        estado = insert(estado, 3, temp[17])
                        estado = insert(estado, 0, temp[18])
                        estado = insert(estado, 1, temp[19])
                    # Back 180
                    if paso_actual == 'B2':
                        temp[0], temp[1], temp[2] = estado[2], estado[1], estado[0]
                        temp[3], temp[4], temp[5] = estado[36], estado[39], estado[42]
                        temp[6], temp[7], temp[8] = estado[33], estado[34], estado[35]
                        temp[9], temp[10], temp[11] = estado[17], estado[14], estado[11]
                        temp[12], temp[13], temp[14] = estado[45], estado[46], estado[47]
                        temp[15], temp[16], temp[17] = estado[50], estado[53], estado[52]
                        temp[18], temp[19] = estado[51], estado[48]

                        estado = insert(estado, 36, temp[0])
                        estado = insert(estado, 39, temp[1])
                        estado = insert(estado, 42, temp[2])
                        estado = insert(estado, 33, temp[3])
                        estado = insert(estado, 34, temp[4])
                        estado = insert(estado, 35, temp[5])
                        estado = insert(estado, 17, temp[6])
                        estado = insert(estado, 14, temp[7])
                        estado = insert(estado, 11, temp[8])
                        estado = insert(estado, 2, temp[9])
                        estado = insert(estado, 1, temp[10])
                        estado = insert(estado, 0, temp[11])
                        estado = insert(estado, 47, temp[12])
                        estado = insert(estado, 50, temp[13])
                        estado = insert(estado, 53, temp[14])
                        estado = insert(estado, 52, temp[15])
                        estado = insert(estado, 51, temp[16])
                        estado = insert(estado, 48, temp[17])
                        estado = insert(estado, 45, temp[18])
                        estado = insert(estado, 46, temp[19])

                        temp[0], temp[1], temp[2] = estado[2], estado[1], estado[0]
                        temp[3], temp[4], temp[5] = estado[36], estado[39], estado[42]
                        temp[6], temp[7], temp[8] = estado[33], estado[34], estado[35]
                        temp[9], temp[10], temp[11] = estado[17], estado[14], estado[11]
                        temp[12], temp[13], temp[14] = estado[45], estado[46], estado[47]
                        temp[15], temp[16], temp[17] = estado[50], estado[53], estado[52]
                        temp[18], temp[19] = estado[51], estado[48]

                        estado = insert(estado, 36, temp[0])
                        estado = insert(estado, 39, temp[1])
                        estado = insert(estado, 42, temp[2])
                        estado = insert(estado, 33, temp[3])
                        estado = insert(estado, 34, temp[4])
                        estado = insert(estado, 35, temp[5])
                        estado = insert(estado, 17, temp[6])
                        estado = insert(estado, 14, temp[7])
                        estado = insert(estado, 11, temp[8])
                        estado = insert(estado, 2, temp[9])
                        estado = insert(estado, 1, temp[10])
                        estado = insert(estado, 0, temp[11])
                        estado = insert(estado, 47, temp[12])
                        estado = insert(estado, 50, temp[13])
                        estado = insert(estado, 53, temp[14])
                        estado = insert(estado, 52, temp[15])
                        estado = insert(estado, 51, temp[16])
                        estado = insert(estado, 48, temp[17])
                        estado = insert(estado, 45, temp[18])
                        estado = insert(estado, 46, temp[19])
                    # Left 180
                    if paso_actual == 'L2':
                        temp[0], temp[1], temp[2] = estado[0], estado[3], estado[6]
                        temp[3], temp[4], temp[5] = estado[18], estado[21], estado[24]
                        temp[6], temp[7], temp[8] = estado[27], estado[30], estado[33]
                        temp[9], temp[10], temp[11] = estado[53], estado[50], estado[47]
                        temp[12], temp[13], temp[14] = estado[36], estado[37], estado[38]
                        temp[15], temp[16], temp[17] = estado[41], estado[44], estado[43]
                        temp[18], temp[19] = estado[42], estado[39]

                        estado = insert(estado, 18, temp[0])
                        estado = insert(estado, 21, temp[1])
                        estado = insert(estado, 24, temp[2])
                        estado = insert(estado, 27, temp[3])
                        estado = insert(estado, 30, temp[4])
                        estado = insert(estado, 33, temp[5])
                        estado = insert(estado, 53, temp[6])
                        estado = insert(estado, 50, temp[7])
                        estado = insert(estado, 47, temp[8])
                        estado = insert(estado, 0, temp[9])
                        estado = insert(estado, 3, temp[10])
                        estado = insert(estado, 6, temp[11])
                        estado = insert(estado, 38, temp[12])
                        estado = insert(estado, 41, temp[13])
                        estado = insert(estado, 44, temp[14])
                        estado = insert(estado, 43, temp[15])
                        estado = insert(estado, 42, temp[16])
                        estado = insert(estado, 39, temp[17])
                        estado = insert(estado, 36, temp[18])
                        estado = insert(estado, 37, temp[19])

                        temp[0], temp[1], temp[2] = estado[0], estado[3], estado[6]
                        temp[3], temp[4], temp[5] = estado[18], estado[21], estado[24]
                        temp[6], temp[7], temp[8] = estado[27], estado[30], estado[33]
                        temp[9], temp[10], temp[11] = estado[53], estado[50], estado[47]
                        temp[12], temp[13], temp[14] = estado[36], estado[37], estado[38]
                        temp[15], temp[16], temp[17] = estado[41], estado[44], estado[43]
                        temp[18], temp[19] = estado[42], estado[39]

                        estado = insert(estado, 18, temp[0])
                        estado = insert(estado, 21, temp[1])
                        estado = insert(estado, 24, temp[2])
                        estado = insert(estado, 27, temp[3])
                        estado = insert(estado, 30, temp[4])
                        estado = insert(estado, 33, temp[5])
                        estado = insert(estado, 53, temp[6])
                        estado = insert(estado, 50, temp[7])
                        estado = insert(estado, 47, temp[8])
                        estado = insert(estado, 0, temp[9])
                        estado = insert(estado, 3, temp[10])
                        estado = insert(estado, 6, temp[11])
                        estado = insert(estado, 38, temp[12])
                        estado = insert(estado, 41, temp[13])
                        estado = insert(estado, 44, temp[14])
                        estado = insert(estado, 43, temp[15])
                        estado = insert(estado, 42, temp[16])
                        estado = insert(estado, 39, temp[17])
                        estado = insert(estado, 36, temp[18])
                        estado = insert(estado, 37, temp[19])
                    # Down 180
                    if paso_actual == 'D2':
                        temp[0], temp[1], temp[2] = estado[42], estado[43], estado[44]
                        temp[3], temp[4], temp[5] = estado[24], estado[25], estado[26]
                        temp[6], temp[7], temp[8] = estado[15], estado[16], estado[17]
                        temp[9], temp[10], temp[11] = estado[51], estado[52], estado[53]
                        temp[12], temp[13], temp[14] = estado[27], estado[28], estado[29]
                        temp[15], temp[16], temp[17] = estado[32], estado[35], estado[34]
                        temp[18], temp[19] = estado[33], estado[30]

                        estado = insert(estado, 24, temp[0])
                        estado = insert(estado, 25, temp[1])
                        estado = insert(estado, 26, temp[2])
                        estado = insert(estado, 15, temp[3])
                        estado = insert(estado, 16, temp[4])
                        estado = insert(estado, 17, temp[5])
                        estado = insert(estado, 51, temp[6])
                        estado = insert(estado, 52, temp[7])
                        estado = insert(estado, 53, temp[8])
                        estado = insert(estado, 42, temp[9])
                        estado = insert(estado, 43, temp[10])
                        estado = insert(estado, 44, temp[11])
                        estado = insert(estado, 29, temp[12])
                        estado = insert(estado, 32, temp[13])
                        estado = insert(estado, 35, temp[14])
                        estado = insert(estado, 34, temp[15])
                        estado = insert(estado, 33, temp[16])
                        estado = insert(estado, 30, temp[17])
                        estado = insert(estado, 27, temp[18])
                        estado = insert(estado, 28, temp[19])

                        temp[0], temp[1], temp[2] = estado[42], estado[43], estado[44]
                        temp[3], temp[4], temp[5] = estado[24], estado[25], estado[26]
                        temp[6], temp[7], temp[8] = estado[15], estado[16], estado[17]
                        temp[9], temp[10], temp[11] = estado[51], estado[52], estado[53]
                        temp[12], temp[13], temp[14] = estado[27], estado[28], estado[29]
                        temp[15], temp[16], temp[17] = estado[32], estado[35], estado[34]
                        temp[18], temp[19] = estado[33], estado[30]

                        estado = insert(estado, 24, temp[0])
                        estado = insert(estado, 25, temp[1])
                        estado = insert(estado, 26, temp[2])
                        estado = insert(estado, 15, temp[3])
                        estado = insert(estado, 16, temp[4])
                        estado = insert(estado, 17, temp[5])
                        estado = insert(estado, 51, temp[6])
                        estado = insert(estado, 52, temp[7])
                        estado = insert(estado, 53, temp[8])
                        estado = insert(estado, 42, temp[9])
                        estado = insert(estado, 43, temp[10])
                        estado = insert(estado, 44, temp[11])
                        estado = insert(estado, 29, temp[12])
                        estado = insert(estado, 32, temp[13])
                        estado = insert(estado, 35, temp[14])
                        estado = insert(estado, 34, temp[15])
                        estado = insert(estado, 33, temp[16])
                        estado = insert(estado, 30, temp[17])
                        estado = insert(estado, 27, temp[18])
                        estado = insert(estado, 28, temp[19])
                        
        # glRotatef(1, 1, 1, 1)
        # clear buffers to preset values
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Cube(estado)
        # Update the full display Surface to the screen
        pygame.display.flip()
        # pause the program for an amount of time
        pygame.time.wait(10)
        # window name
        pygame.display.set_caption('Cube Animation')

    # URFDLBURFDLBURFDLBURFDLBURFDLBURFDLBURFDLBURFDLBURFDLB
    # UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
    # DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD
    # BRLFFUBBDDBLLRFRDBLLRBUFRLRLUDRDUBUDUDFRRBFLFURFDBFUDU

    # R' D2 R' U2 R F2 D B2 U' R F' U R2 D L2 D' B2 R2 B2 U' B2

def solve(initial):
    solution = kociemba.solve(initial)
    main(initial, solution)


solve(init_state)