# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# DBM - Command line to create .exe file : pyinstaller .\main.py inside the venv
# DBM - Tesseract v5.0 installed
# DBM - 200 dpi seems optimum & letters might have a font size of 30-33 (pixels) regarding the litterature
# https://groups.google.com/g/tesseract-ocr/c/Wdh_JJwnw94/m/24JHDYQbBQAJ
# DBM -

import os
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
import tkinter
from tkinter import *
from tkinter import filedialog
import pytesseract
from pytesseract import Output
import shutil
try:
    from PIL import Image
except ImportError:
    import Image


def init_var():
    global imagesToPrint
    global idxImagesToPrint
    imagesToPrint = []
    idxImagesToPrint = 1
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ifFolderExistsDeleteAllFilesFromIt('ImagesToPrint')
    ifFolderExistsDeleteAllFilesFromIt('Text')


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
    dirName = 'PagesFromPdfFile'
    folder_exist = os.path.isdir(dirName)
    directoryPagesFromPdfFile = os.path.join(os.getcwd(), dirName)
    if not folder_exist:
        os.mkdir(directoryPagesFromPdfFile)
    fname = directoryPagesFromPdfFile+'\page' + str(i) + '.png'
    png_file.save(fname, 'PNG')

    img = cv2.imread(fname, cv2.IMREAD_COLOR)

    # RGB to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    global trueGray
    trueGray = gray.copy()

    gray = ~gray
    # plotImg(gray))

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


def ifFolderExistsDeleteAllFilesFromIt(folderName):
    folder_exist = os.path.isdir(folderName)
    pathOfFolder = os.path.join(os.getcwd(), folderName)
    if folder_exist:
        shutil.rmtree(pathOfFolder)

    os.mkdir(pathOfFolder)


def mkdir_and_imwrite_selected_images_and_do_pytesseract():
    global idxImagesToPrint

    pathTxt = os.path.join(os.getcwd(), 'Text')

    for i in range(len(imagesToPrint)):

        txtName = pathTxt + '\Text' + str(idxImagesToPrint) + '.csv'
        f = open(txtName, 'w')
        f.write('Ref\n')
        x, y, w, h = imagesToPrint[i]
        tmp_img = trueGray[y:y + h, x:x + w]
        #plotImg(tmp_img)
        res = processImgtoText(tmp_img)
        for j in range(len(res)):
            f.write(res[j][0:-2]+"\n")
        f.close()
        idxImagesToPrint = idxImagesToPrint + 1


def unskew_the_image(img):
    ThreshValue = 100
    img2 = cv2.bitwise_not(img)
    # plotImg(img2, "img2")
    # from here image is black in background and light for the text
    thresh = cv2.threshold(img2, ThreshValue, 255, cv2.THRESH_BINARY)[1]
    # plotImg(thresh, "Thresh")
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    val = np.where(thresh > 0)
    coords = np.column_stack([val[1], val[0]])
    theRect = cv2.minAreaRect(coords)
    # box = cv2.boxPoints(theRect)
    # box = np.int0(box)
    # cv2.drawContours(thresh, [box], 0, (255, 255, 255), 1)
    angle = theRect[-1]
    # rotate the image to deskew it
    # if angle == 90:
    #     angle = 0
    #     plotImg(thresh)
    #     box = cv2.boxPoints(theRect)
    #     box = np.int0(box)
    #     cv2.drawContours(thresh, [box], 0, (255, 255, 255), 1)
    #     plotImg(thresh)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # print("[INFO] angle: {:.3f}".format(angle))
    return rotated


def drawMinAreaRect(img):
    ThreshValue = 160
    img = cv2.bitwise_not(img)
    thresh = cv2.threshold(img, ThreshValue, 255, cv2.THRESH_BINARY)[1]
    val = np.where(thresh > 0)
    coords = np.column_stack([val[1], val[0]])
    theRect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(theRect)
    box = np.int0(box)
    cv2.drawContours(thresh, [box], 0, (0, 0, 0), 1)
    plotImg(thresh)


class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.l=Label(top,text="Hello World")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()


class mainWindow(object):
    def __init__(self,master):
        self.master=master
        self.b=Button(master,text="click me!",command=self.popup)
        self.b.pack()
        self.b2=Button(master,text="print value",command=lambda: sys.stdout.write(self.entryValue()+'\n'))
        self.b2.pack()

    def popup(self):
        self.w=popupWindow(self.master)
        self.b["state"] = "disabled"
        self.master.wait_window(self.w.top)
        self.b["state"] = "normal"

    def entryValue(self):
        return self.w.value


def decideWhatToDoWithTheResults(txt_value_from_complete_ref_box, txt_value_single_letter_box, txt_value_from_all_ref_boxes_from_the_selected_sheet):
    ref_from_complete_ref_box = txt_value_from_complete_ref_box[-4:]
    ref_from_single_letter_box = txt_value_single_letter_box[-4:]
    ref_from_all_ref_boxes_from_the_selected_sheet = txt_value_from_all_ref_boxes_from_the_selected_sheet[-4:]
    #TODO - Warning ! Error when : e.g. S643 | 5643 | 5643 from args... but must not modify the global behaviour
    if "#" in ref_from_complete_ref_box or "#" in ref_from_single_letter_box or "#" in ref_from_all_ref_boxes_from_the_selected_sheet \
        or ref_from_complete_ref_box[0].isalpha() or ref_from_single_letter_box[0].isalpha() or ref_from_all_ref_boxes_from_the_selected_sheet[0].isalpha() :
        if txt_value_from_complete_ref_box == txt_value_single_letter_box:
            return txt_value_from_complete_ref_box
        elif txt_value_from_complete_ref_box == txt_value_from_all_ref_boxes_from_the_selected_sheet:
            return txt_value_from_complete_ref_box
        elif txt_value_single_letter_box == txt_value_from_all_ref_boxes_from_the_selected_sheet:
            return txt_value_single_letter_box
        else:# All three disagree...
            root = Tk()
            m = mainWindow(root)
            root.mainloop()
            a=2
    else:# ref with a letter first (or something else...)
        if ref_from_complete_ref_box[0].isalpha() or ref_from_single_letter_box[0].isalpha() or ref_from_all_ref_boxes_from_the_selected_sheet[0].isalpha():
            if ref_from_complete_ref_box[0].isalpha():# To continue...
                return 'FUCK !'
        else:
            return "???"# Display everything because this is a mess ! :)


def processImgtoText(img):
    minimumSizeOfARef = 10  # A02B-XXXX- = 10 char (4+1+4+1)
    padding_size = 10 # 10 pixels around text
    img = cv2.GaussianBlur(img, (5, 5), 0)# test for better recognition of the reference
    # plotImg(img, "the Blured Image")
    # Separate all of the text boxes for better reading of the OCR || psm=11 not so bad || psm=12 perfect !

    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 12')
    # uncomment for debugging
    # n_boxes = len(d['level'])
    # theImage = img.copy()
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(theImage, (x, y), (x + w, y + h), (0, 0, 0), 1)
    # plotImg(theImage, "the Image with Boxes")

    n_boxes = len(d['level'])
    res = []
    txt_value_from_all_ref_boxes_from_the_selected_sheet = ""
    for i in range(n_boxes):
        # centers = []
        if d["text"][i] != "" and len(d["text"][i]) > minimumSizeOfARef:
            txt_value_from_all_ref_boxes_from_the_selected_sheet = d["text"][i]
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img_bordered = img[y:y + h, x:x + w]

            # Padding of the image for better results of the OCR
            img_padded = cv2.copyMakeBorder(img_bordered, padding_size, padding_size, padding_size, padding_size,
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])

            # plotImg(img_padded)

            #Unskew the text
            img_unskewed = unskew_the_image(img_padded)
            # plotImg(img_unskewed)

            # From there image the text is straight

            #Blur the image with a Gaussian filter of 5x5
            img_unskewed_and_blured = cv2.GaussianBlur(img_unskewed, (5, 5), 0)

            # In order to get another result for comparing
            txt_value_single_letter_boxe = ""
            boxes = pytesseract.image_to_boxes(img_unskewed_and_blured, config='--psm 10 --oem 1 -c tessedit_char_whitelist=#-ABEFHJRSMTKVNG0123456789')
            for b in boxes.splitlines():
                b = b.split(' ')
                txt_value_single_letter_boxe = txt_value_single_letter_boxe + b[0]

            # plotImg(tmp_img, "tmp_img")
            # psm=13 ng | psm=8 ng | psm=7 good | psm=6 ng | psm=4 ng
            txt_value_from_complete_ref_box = pytesseract.image_to_string(img_unskewed_and_blured, config='--psm 7 --oem 1 -c tessedit_char_whitelist=#-ABEFHJRSMTKVNG0123456789')
            txt_value_from_complete_ref_box = txt_value_from_complete_ref_box.rstrip()
            if txt_value_from_complete_ref_box != txt_value_single_letter_boxe \
                or txt_value_single_letter_boxe != txt_value_from_all_ref_boxes_from_the_selected_sheet \
                or txt_value_from_all_ref_boxes_from_the_selected_sheet != txt_value_from_complete_ref_box:
                print("Différent ! : " + txt_value_from_complete_ref_box.rstrip() + " || " + txt_value_single_letter_boxe + " || " + txt_value_from_all_ref_boxes_from_the_selected_sheet)
                txt_to_append = decideWhatToDoWithTheResults(txt_value_from_complete_ref_box, txt_value_single_letter_boxe, txt_value_from_all_ref_boxes_from_the_selected_sheet)
            else:
                txt_to_append = txt_value_from_complete_ref_box
            res.append(txt_to_append)
    return res


def extract_reference_and_create_excel_file():
    directory = "./Text"
    listOfFiles = os.listdir(directory)  # dir is your directory path
    number_files = len(listOfFiles)
    workbook = xlsxwriter.Workbook(SaveExcelFile())

    worksheetHardware = workbook.add_worksheet('Hardware')
    worksheetOptions = workbook.add_worksheet('Options')
    cpt = 0
    worksheetHardware.write(cpt, 0, "Référence")
    worksheetHardware.write(cpt, 1, "Max Value Assembly")
    worksheetOptions.write(cpt, 0, "Référence")
    worksheetOptions.write(cpt, 1, "FROM")
    worksheetOptions.write(cpt, 2, "SRAM")
    worksheetOptions.write(cpt, 3, "DRAM")
    cpt = 1

    for i in range(number_files):  # loop for all txt
        line = pd.read_csv("./Text/Text" + str(i + 1) + ".csv")
        for completeRef in line.values:
            EndOfRef = completeRef[0].replace(" ", "").split('-')[-1]
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
