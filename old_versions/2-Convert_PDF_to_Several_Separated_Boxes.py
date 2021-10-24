# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plotImg(img, title=''):
    plt.imshow(img, cmap='Greys_r')
    plt.title(title)
    plt.show()


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


if __name__ == '__main__':
    images = convert_from_path('example.pdf')

    for i in range(len(images)):
        fname = 'page' + str(i) + '.png'
        images[i].save(fname, 'PNG')

        # TODO coder la déctection des boxes dans l'image
        img = cv2.imread(fname)
        # RGB to Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = ~gray
        # plotImg(gray)

        # Dilation of the image
        kernel = np.ones((5, 5), np.uint8)
        img_closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # plotImg(img_closing)

        ret, threshed_image = cv2.threshold(img_closing, 10, 255, cv2.THRESH_BINARY)
        #plotImg(threshed_image, "Threshed Image")

        # Now we have complete closed boxes in the image --> We can detect the boxes now with morphological operations
        # We are defining Kernels for Vertical and Horizontal lines detection and indexation
        kernel_length = np.array(img).shape[1] // 80
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Detection of Vertical lines
        img_temp1 = cv2.erode(threshed_image, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
        #plotImg(verticle_lines_img, "vertical Lines")

        # Detection of Horizontal lines
        img_temp2 = cv2.erode(threshed_image, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        #plotImg(horizontal_lines_img, "horizontal Lines")

        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha
        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #plotImg(img_final_bin, 'Image après somme des deux masks')

        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort all the contours by top to bottom.
        contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

        idx = 0
        result_dir_path = "./results"
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            if (w > 190 and w < 1500 ) and (h > 45 and h < 1500):
                idx += 1
                new_img = img[y:y + h, x:x + w]
                imgName = 'Box détectée, w = ' + str(w) + ' & h = ' + str(h)
                #plotImg(new_img, imgName)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
                #cv2.imwrite(result_dir_path + str(idx) + '.png', new_img)
        plotImg(img, 'Avec Contour')
