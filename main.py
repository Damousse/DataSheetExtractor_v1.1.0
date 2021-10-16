# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# DBM - Command line to create .exe file : pyinstaller .\main.py inside the venv

import os
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
import re
import tkinter
from tkinter import filedialog


try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def init_var():
    global imagesToPrint
    global idxImagesToPrint
    imagesToPrint = []
    idxImagesToPrint = 1
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def getPdfFile():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    currdir = os.getcwd()
    return filedialog.askopenfilename(parent=root,
                                      initialdir=currdir,
                                      title='Please select a File')


def SaveExcelFile():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    currdir = os.getcwd()
    return filedialog.asksaveasfilename(parent=root,
                                        initialdir=currdir,
                                        title='Please select a destination',
                                        initialfile="Results.xlsx",
                                        defaultextension=".xlsx",
                                        filetypes=[("Excel files", '*.xlsx')])


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


def detect_boxes_in_the_png_file(png_file):
    fname = 'page' + str(i) + '.png'
    png_file.save(fname, 'PNG')

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
    # plotImg(threshed_image, "Threshed Image")

    # Now we have complete closed boxes in the image --> We can detect the boxes now with morphological operations
    # We are defining Kernels for Vertical and Horizontal lines detection and indexation
    kernel_length = np.array(img).shape[1] // 80
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Detection of Vertical lines
    img_temp1 = cv2.erode(threshed_image, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # plotImg(verticle_lines_img, "vertical Lines")

    # Detection of Horizontal lines
    img_temp2 = cv2.erode(threshed_image, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    # plotImg(horizontal_lines_img, "horizontal Lines")

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function adds two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # plotImg(img_final_bin, 'Image après somme des deux masks')

    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingboxes = sort_contours(contours, method="top-to-bottom")

    idx = 0
    trueContours = []
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        if (190 < w < 1500) and (45 < h < 1850):
            # idx += 1
            # new_img = img[y:y + h, x:x + w]
            # imgName = 'Box détectée, w = ' + str(w) + ' & h = ' + str(h)
            # cv2.imwrite(imgName, new_img)
            # plotImg(new_img, imgName)
            trueContours.append(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # plotImg(img, 'Avec Contour')
    return img, trueContours


# Callback method to know if we will use or not the selected region
# Right click --> Disable // Left click --> Enable
def mouse_event(event, x, y, flags, param):
    for contour in param:
        result = cv2.pointPolygonTest(contour, (x, y), False)
        xc, yc, w, h = cv2.boundingRect(contour)
        if event == cv2.EVENT_LBUTTONDOWN and result >= 0:  # Change the colours of the contour : Green to Blue
            cv2.rectangle(img, (xc, yc), (xc + w, yc + h), (255, 0, 0), 3)
            imagesToPrint.append([xc, yc, w, h])
        elif event == cv2.EVENT_RBUTTONDOWN and result >= 0:  # Change the colours of the contour : Blue to Green
            cv2.rectangle(img, (xc, yc), (xc + w, yc + h), (0, 255, 0), 3)
            if imagesToPrint.count([xc, yc, w, h]) > 0:
                imagesToPrint.remove([xc, yc, w, h])

        cv2.imshow('Current image', img)


def select_box_by_click(img, truecontours):
    cv2.namedWindow('Current image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Current image', mouse_event, truecontours)
    img = cv2.resize(img, (960, 540))
    cv2.imshow('Current image', img)

    # Wait for user to press enter and then continue
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def mkdir_and_imwrite_selected_images_and_do_pytesseract():
    global idxImagesToPrint
    folder_exist = os.path.isdir('ImagesToPrint')
    pathImages = os.path.join(os.getcwd(), 'ImagesToPrint')
    if not folder_exist:
        os.mkdir(pathImages)
    folder_exist = os.path.isdir('Text')
    pathTxt = os.path.join(os.getcwd(), 'Text')
    if not folder_exist:
        os.mkdir(pathTxt)

    for i in range(len(imagesToPrint)):
        x, y, w, h = imagesToPrint[i]
        tmp_img = img[y:y + h, x:x + w]
        imgName = pathImages + '\Image' + str(idxImagesToPrint) + '.png'
        # Blurring the image so that Pytesseract can detect char better
        tmp_img2 = cv2.GaussianBlur(tmp_img, (3, 3), 0)
        cv2.imwrite(imgName, tmp_img2)
        res = pytesseract.image_to_string(Image.open(imgName), config='--oem 3 --psm 4')
        print('Res : \n' + res)
        txtName = pathTxt + '\Text' + str(idxImagesToPrint) + '.csv'
        f = open(txtName, 'w')
        f.write(res)
        f.close()
        idxImagesToPrint = idxImagesToPrint + 1

        # TODO - 2 - regarder le site car l'OCR basique ne marche pas tout le temps... Essayer de récupérer avec du Custom Deep Learning
        # https://nanonets.com/blog/attention-ocr-for-text-recogntion/


def extract_reference_and_create_excel_file():
    directory = "./Text"
    listOfFiles = os.listdir(directory)  # dir is your directory path
    number_files = len(listOfFiles)
    workbook = xlsxwriter.Workbook(SaveExcelFile())

    worksheetHardware = workbook.add_worksheet('Hardware')
    worksheetOptions = workbook.add_worksheet('Options')
    cpt = 0
    worksheetHardware.write(cpt,0,"Référence")
    worksheetHardware.write(cpt, 1, "Max Value Assembly")
    worksheetOptions.write(cpt,0,"Référence")
    worksheetOptions.write(cpt,1,"FROM")
    worksheetOptions.write(cpt,2,"SRAM")
    worksheetOptions.write(cpt,3,"DRAM")
    cpt = 1

    for i in range(number_files):  # loop for all txt
        line = pd.read_csv("./Text/Text" + str(i + 1) + ".csv")
        for completeRef in line.values:
            completeRef[0].replace(" ", "")
            EndOfRef = completeRef[0].split('-')[-1]
            if EndOfRef != "" and len(EndOfRef) > 3:
                if i == 0:  # hardware CNC
                    worksheetHardware.write(cpt, 0, EndOfRef)
                    cpt = cpt + 1
                else:  # Options
                    worksheetOptions.write(cpt, 0, EndOfRef)
                    cpt = cpt + 1
        if i == 0 and cpt > 0:
            cpt = 1
    workbook.close()


if __name__ == '__main__':
    pages = convert_from_path(getPdfFile())
    init_var()
    for i in range(len(pages)):
        img, trueContours = detect_boxes_in_the_png_file(pages[i])

        # Récupérer le pointeur de la souris et trouver le contour le plus près suite à click sur l'image ?
        select_box_by_click(img, trueContours)

        # imagesToPrint containt the rectangle for image analysis
        # Do also pytesseract
        mkdir_and_imwrite_selected_images_and_do_pytesseract()

        imagesToPrint.clear()

    extract_reference_and_create_excel_file()
