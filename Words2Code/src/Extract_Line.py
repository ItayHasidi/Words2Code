import copy
import queue

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

PATH = "../resources/in"
PATH_OUT = "../resources/out"
MIN_LETTER_HEIGHT = 15
MIN_LETTER_WIDTH = 15
MIN_LETTER_VALUE = 5
MIN_LETTER_WIDTH_VALUE = 30
DIVIDER_HEIGHT = 0.1
DIVIDER_WIDTH = 0.1
FINAL_SIZE = 64
PIXEL_RADIUS = 4
PIXEL_HEIGHT_RADIUS = 6

"""
Extracts letters line by line and converts it to text.
"""


# Saves an image to a file
def save_img(img, name="l", add_ending: bool = True):
    img2 = Image.fromarray(np.array(img), 'L')
    if add_ending:
        img2.save(PATH_OUT + name + ".jpeg")
    else:
        img2.save(PATH_OUT + name)


# Gets an image and transposes each cell to its negative value.
# Black turns to white and white turns to black.
def convert_to_black(img_arr):
    img_out = copy.deepcopy(img_arr)
    for i in range(len(img_arr)):
        for j in range(len(img_arr[0])):
            temp_num = 255 - img_arr[i][j]
            img_out[i][j] = temp_num
    save_img(img_out, "blackImg")
    return img_out


# Converts an image to an 2D array.
def img_to_array(path):
    img = Image.open(path)
    img_arr = np.asarray(img.convert('L'))
    img_out = copy.deepcopy(img_arr)
    return img_out


# Splits the image according to its text lines.
def seperate_lines(image):
    lines = []
    n = len(image)
    m = len(image[0])
    i = 0
    # separates the lines
    while i in range(n):
        line = [0, 0]  # pos_0 - start of the line, pos_1 - end of the line
        if image[i][0] > MIN_LETTER_VALUE:
            line[0] = i
            while (i + 6 < n) and (image[i][0] > MIN_LETTER_VALUE
                                   or image[i + 1][0] > MIN_LETTER_VALUE or image[i + 2][0] > MIN_LETTER_VALUE
                                   or image[i + 3][0] > MIN_LETTER_VALUE or image[i + 4][0] > MIN_LETTER_VALUE
                                   or image[i + 5][0] > MIN_LETTER_VALUE or image[i + 6][0] > MIN_LETTER_VALUE):
                i += 1
            if i - line[0] > MIN_LETTER_HEIGHT:
                line[1] = i
                lines.append(line)
        i += 1
    return lines


# Finds all the letters for each line.
def find_letter(image, visited_img, start_j):
    height = len(image)
    width = len(image[0])
    # q = []
    j = start_j
    i = 0
    while j < width:
        i = 0
        while i < height:
            # print(j, i)

            if image[i][j] > MIN_LETTER_VALUE * 6 and visited_img[i][j] != 1:
                # print("found")
                visited_img[i][j] = 1
                # q.append([i, j, image[i][j]])  # index_i, index_j, value at indexes
                return i, j
            visited_img[i][j] = 1
            i += 1
        j += 1
    return i, j


# def find_neighbor_cells(image, visited_img, start_j):
#     height = len(image)
#     width = len(image[0])
#     min_pixel = [height, width]
#     max_pixel = [0, 0]
#     good_pixels = []
#     q = []
#     j = start_j
#     i = 0
#     while j < width:
#         i = 0
#         while i < height:
#             # print(j, i)
#             if image[i][j] > MIN_LETTER_VALUE * 3 and visited_img[i][j] != 1:
#                 # print("found")
#                 visited_img[i][j] = 1
#                 q.append([i, j])
#                 good_pixels.append([i, j])
#                 # q.append([i, j, image[i][j]])  # index_i, index_j, value at indexes
#                 while len(q) > 0:
#                     c_i, c_j = q.pop()
#                     if c_i < min_pixel[0]:
#                         min_pixel[0] = c_i
#                     if c_j < min_pixel[1]:
#                         min_pixel[1] = c_j
#                     if c_i > max_pixel[0]:
#                         max_pixel[0] = c_i
#                     if c_j > max_pixel[1]:
#                         max_pixel[1] = c_j
#                     for k in range(PIXEL_RADIUS):
#                         for l in range(PIXEL_RADIUS):
#                             if c_i + k < height and c_j + l < width:
#                                 if image[c_i + k][c_j + l] > MIN_LETTER_VALUE * 3 \
#                                         and visited_img[c_i + k][c_j + l] == 0:
#                                     q.append([c_i + k, c_j + l])
#                                     good_pixels.append([c_i, c_j])
#                                 visited_img[c_i + k][c_j + l] = 1
#
#                 return good_pixels, min_pixel, max_pixel, c_j
#             visited_img[i][j] = 1
#             i += 1
#         j += 1
#     return good_pixels, min_pixel, max_pixel, j


# def seperate_letters(image: list, name):
#     words = []
#     letter_counter = 0
#     counter = 0
#     saved_pixels: list = []
#     height = len(image)
#     width = len(image[0])
#     visited_img = np.zeros([height, width])
#
#     # min_pixel = [height, height]
#     # max_pixel = [0, 0]
#     c_j = 0
#     while c_j < width:
#         q, min_pixel, max_pixel, c_j = find_neighbor_cells(image, visited_img, 1)
#         # cur_i, cur_j = q.pop()
#         # if max_pixel[0] - min_pixel[0] > MIN_LETTER_HEIGHT and max_pixel[1] - min_pixel[1] > MIN_LETTER_WIDTH:
#         height_img = int((max_pixel[0] - min_pixel[0]) * 1)
#         width_img = int((max_pixel[1] - min_pixel[1]) * 1)
#         if height_img > MIN_LETTER_HEIGHT and width_img > MIN_LETTER_WIDTH:
#             height_img_offset = height_img - (max_pixel[0] - min_pixel[0])
#             width_img_offset = width_img - (max_pixel[1] - min_pixel[1])
#             res_img = np.zeros([height_img, width_img])
#
#             name_string = "/"
#             if -1 < name < 10:
#                 name_string += "0"
#             name_string += str(name) + "_"
#             if -1 < counter < 10:
#                 name_string += "0"
#             name_string += str(counter) + ".jpeg"
#             save_img(res_img, name_string, False)
#             res_img = img_to_array(PATH_OUT + name_string)
#
#             for cell in q:
#                 c_i, c_j = cell
#                 res_img[c_i - min_pixel[0] - 1][c_j - min_pixel[1] - 1] = image[c_i][c_j]
#
#             save_img(res_img, name_string, False)
#             counter += 1


#
# """
def seperate_letters(image: list, name):
    words = []
    letter_counter = 0
    counter = 0
    saved_pixels: list = []
    height = len(image)
    width = len(image[0])
    min_pixel = [height, height]
    max_pixel = [0, 0]
    visited_img = np.zeros([height, width])
    cur_i, cur_j = find_letter(image, visited_img, 1)
    q = []
    q.append([cur_i, cur_j])
    # cur_j = (res[0])[1]
    while cur_j < width:

        while len(q) > 0:
            cell = q.pop()
            i = cell[0]
            j = cell[1]
            # val = cell[2]
            # print(i, j, val)
            if i < min_pixel[0]:
                min_pixel[0] = i
            if j < min_pixel[1]:
                min_pixel[1] = j
            if i > max_pixel[0]:
                max_pixel[0] = i
            if j > max_pixel[1]:
                max_pixel[1] = j
            temp_i = 0
            temp_j = 1
            visited_img[i, j] = 1
            if i - PIXEL_HEIGHT_RADIUS > 0:
                temp_i = i - PIXEL_HEIGHT_RADIUS
            if j - PIXEL_RADIUS > 1:
                temp_j = j - PIXEL_RADIUS
            if image[i][j] > MIN_LETTER_VALUE:
                if not saved_pixels.__contains__([i, j]):
                    saved_pixels.append([i, j])
                    # saved_pixels.sort()

                for n in range(PIXEL_HEIGHT_RADIUS * 2):
                    for m in range(PIXEL_RADIUS * 2):
                        if temp_i + n < height and temp_j + m < width and visited_img[temp_i + n][temp_j + m] == 0:
                            q.append([temp_i + n, temp_j + m, image[temp_i + n][temp_j + m]])

        if max_pixel[0] - min_pixel[0] > MIN_LETTER_HEIGHT and max_pixel[1] - min_pixel[1] > MIN_LETTER_WIDTH:
            height_img = int((max_pixel[0] - min_pixel[0]) * 1)
            width_img = int((max_pixel[1] - min_pixel[1]) * 1)
            height_img_offset = height_img - (max_pixel[0] - min_pixel[0])
            width_img_offset = width_img - (max_pixel[1] - min_pixel[1])

            res_img = np.zeros([height_img, width_img])
            if -1 < counter < 10:
                # save_img(res_img, "/" + str(name) + "Q_0" + str(counter))
                save_img(res_img, "/" + str(name) + "_0" + str(counter))
                res_img = img_to_array(PATH_OUT + "/" + str(name) + "_0" + str(counter) + ".jpeg")
            else:
                # save_img(res_img, "/" + str(name) + "Q_" + str(counter))
                save_img(res_img, "/" + str(name) + "_" + str(counter))
                res_img = img_to_array(PATH_OUT + "/" + str(name) + "_" + str(counter) + ".jpeg")

            # for idx in saved_pixels:
            #     res_img[idx[0], idx[1]] = image[idx[0]][idx[1]]
            for colored_cell in saved_pixels:
                if 0 < colored_cell[0] - min_pixel[0] + height_img_offset / 2 < height_img \
                        and 0 < colored_cell[1] - min_pixel[1] + width_img_offset / 2 < width_img:
                    res_img[colored_cell[0] - min_pixel[0] + int(height_img_offset / 2)] \
                [colored_cell[1] - min_pixel[1] + int(width_img_offset / 2)] = image[colored_cell[0]][colored_cell[1]]
            if -1 < counter < 10:
                save_img(res_img, "/" + str(name) + "_0" + str(counter))
            else:
                save_img(res_img, "/" + str(name) + "_" + str(counter))
            counter += 1
        # cur_j = max_pixel[1]

        # max_pixel = [-1, -1]
        saved_pixels.clear()
        # q.append([0, min_pixel[1]])
        cur_i, cur_j = find_letter(image, visited_img, max_pixel[1])
        # min_pixel[0] = max_pixel[0]
        # min_pixel[1] = PIXEL_RADIUS + max_pixel[1]
        min_pixel[0] = cur_j
        min_pixel[1] = cur_j
        q.append([cur_i, cur_j])
        # cur_j = res[1]
    return words


# Handles the splitting of lines and letters.


def get_word(image):
    words = []
    lines = []
    n = len(image)
    m = len(image[0])
    temp_sum = 0
    # makes avg of lines
    f = open(PATH_OUT + "log_values.txt", "w")
    for i in range(n):
        for j in range(m):
            temp_sum += (image[i][j])
        image[i][0] = temp_sum / (n * DIVIDER_HEIGHT)
        f.write(str(i) + ": " + str(image[i][0]) + "\n")
        temp_sum = 0
    f.close()

    lines_idx = seperate_lines(image)
    new_img = []
    # idx = 0
    for k in range(len(lines_idx)):
        i = lines_idx[k][0]
        while i in range(lines_idx[k][1]):
            temp_line = []
            for j in range(m):
                temp_line.append(image[i][j])
            new_img.append(temp_line)
            i += 1
        save_img(new_img, "line_num_" + str(k))
        print("pic num: " + str(k) + ", height: " + str(len(new_img)) + ", width: " + str(len(new_img[0])))
        new_img_temp = seperate_letters(new_img, k)  # str(k) + "_" + str(idx))
        # idx += 1
        words.append(new_img_temp)
        start_line = lines_idx[k][0]
        end_line = lines_idx[k][1]
        new_img.clear()

        img_counter = 0
        for col_width in new_img_temp:
            temp_img = []
            line_idx = start_line
            while line_idx < end_line:
                start_col = col_width[0]
                end_col = col_width[1]
                col_idx = start_col
                temp_holder = []
                while col_idx < end_col:
                    temp_holder.append(image[line_idx][col_idx])
                    col_idx += 1
                temp_img.append(copy.deepcopy(temp_holder))
                temp_holder.clear()
                line_idx += 1
            save_img(temp_img, PATH_OUT + "/" + str(k) + "_" + str(img_counter))
            temp_img.clear()
            img_counter += 1


# Changes the size of the rectangle image to 28x28, so that it would fit the database comparisons.
def resize_to_28(img, filename, name):
    # name = 0
    # filename = filename[:4]
    length = len(img)
    img_28 = np.zeros([FINAL_SIZE, FINAL_SIZE])
    # save_img(img_28, "/" + str(filename), False)
    # img_28 = img_to_array(PATH_OUT + "/" + str(filename))
    incrementer = int(length / FINAL_SIZE) + 1
    i = 0
    j = 0
    while i in range(length):
        while j in range(length):
            pixel_sum = 0
            for n in range(incrementer):
                for m in range(incrementer):
                    if i + n < length and j + m < length:
                        pixel_sum += img[i + n][j + m]
            avg_pixel = int(pixel_sum / (incrementer * incrementer))
            row_idx = int(i / incrementer)
            col_idx = int(j / incrementer)
            if row_idx < FINAL_SIZE and col_idx < FINAL_SIZE:
                img_28[row_idx][col_idx] = avg_pixel
            j += incrementer
        j = 0
        i += incrementer
    i = 0

    # img_28_rotated = np.zeros([FINAL_SIZE, FINAL_SIZE])
    # save_img(img_28_rotated, "/" + str(name), True)
    # img_28_rotated = img_to_array(PATH_OUT + "/" + str(name)+".jpeg")
    # for i in range(FINAL_SIZE):
    #     for j in range(FINAL_SIZE):
    #         img_28_rotated[i][j] = img_28[j][FINAL_SIZE - 1 - i]
    #
    # save_img(img_28, "/" + str(filename), False)
    # name += 1


# Changes the size of the image so that it would be a rectangle.
def resize_img():
    directory = PATH_OUT
    name = 0

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            img = img_to_array(f)
            height = len(img)
            width = len(img[0])
            # Resizing the file so that would be a rectangle
            if height > width:
                rectangle_img = np.zeros([height, height])
                save_img(rectangle_img, "/" + filename, False)
                rectangle_img = img_to_array(PATH_OUT + "/" + filename)
                start_idx = int((height - width) / 2)
                end_idx = start_idx + width
                for i in range(height):
                    j = start_idx
                    k = 0
                    while j in range(end_idx):
                        if img[i][k] > 200:
                            img[i][k] = 250
                        rectangle_img[i][j] = img[i][k]
                        j += 1
                        k += 1
                save_img(rectangle_img, "/" + filename, False)
                img = img_to_array(PATH_OUT + "/" + filename)

            elif width > height:
                rectangle_img = np.zeros([width, width])
                save_img(rectangle_img, "/" + filename, False)
                rectangle_img = img_to_array(PATH_OUT + "/" + filename)
                start_idx = int((width - height) / 2)
                end_idx = start_idx + height
                i = start_idx
                k = 0
                while i in range(end_idx):
                    for j in range(width):
                        if img[k][j] > 200:
                            img[k][j] = 250
                        rectangle_img[i][j] = img[k][j]
                    i += 1
                    k += 1
                save_img(rectangle_img, "/" + filename, False)
                img = img_to_array(PATH_OUT + "/" + filename)
            resize_to_28(img, filename, name)
            name += 1


def main():
    img_arr = img_to_array(PATH + "/avg_func.jpg")  # line_num_0.jpeg   avg_func.jpeg     5_11.jpeg
    img = convert_to_black(img_arr)
    get_word(img)
    resize_img()


if __name__ == "__main__":
    main()
