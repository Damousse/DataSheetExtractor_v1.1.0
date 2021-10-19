# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# DBM - Command line to create .exe file : pyinstaller .\main.py inside the venv
# DBM - Tesseract v5.0 installed on Windows
# DBM - 200 dpi seems optimum & letters might have a font size of 30-33 (pixels) regarding the litterature
# DBM - Python v=3.7 (venv)
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
import imutils
import shutil
try:
    from PIL import Image, ImageTk
except ImportError:
    import Image


def init_var():
    global imagesToPrint
    imagesToPrint = []
    global idxImagesToPrint
    idxImagesToPrint = 1
    # Pytesseract command line
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Functions fro creating all temp folder
    ifFolderExistsDeleteAllFilesFromIt('ImagesToPrint')
    ifFolderExistsDeleteAllFilesFromIt('Text')
    ifFolderExistsDeleteAllFilesFromIt('PagesFromPdfFile')


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


def ifFolderExistsDeleteAllFilesFromIt(folderName):
    folder_exist = os.path.isdir(folderName)
    pathOfFolder = os.path.join(os.getcwd(), folderName)
    if folder_exist:
        shutil.rmtree(pathOfFolder)

    os.mkdir(pathOfFolder)


# it is working
def selectROI(tmp):
    fromCenter = False
    showCrosshair = True
    regions = []
    h, w = tmp.shape[:2]
    # ratio = h_final/775
    ratio = int(h/900)
    tmp2 = cv2.resize(tmp, (int(w/ratio), int(h/ratio)))
    while True:
        ROI = cv2.selectROI('Selector', tmp2, showCrosshair, fromCenter)
        (x, y, w, h) = ROI
        cv2.rectangle(tmp2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        regions.append(tmp[int(ratio * y):int(ratio * (y + h)), int(ratio * x):int(ratio * (x + w))])
        if(ROI[0] == ROI[1]  == ROI[2] == ROI[3] == 0):
            break
    regions.pop()# remove the last one because it is a dummy regions when quiting the 'Selector' view
    cv2.destroyWindow('Selector')
    # for r in regions:
        # plotImg(r)
    return regions


def select_boxes_in_the_png_file(png_file):
    directoryPagesFromPdfFile = os.path.join(os.getcwd(), 'PagesFromPdfFile')
    fname = directoryPagesFromPdfFile+'\page' + str(i) + '.png'
    png_file.save(fname, 'PNG')
    regions_grayed = []

    img = cv2.imread(fname, cv2.IMREAD_COLOR)

    # draw boxes then push enter again and again then push escape when finished
    regions = selectROI(img.copy())

    # RGB to Gray
    for r in regions:
        regions_grayed.append(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))

    return regions_grayed


def mkdir_and_imwrite_selected_images_and_do_pytesseract(images):
    global idxImagesToPrint

    pathTxt = os.path.join(os.getcwd(), 'Text')

    for tmp_img in images:

        txtName = pathTxt + '\Text' + str(idxImagesToPrint) + '.csv'
        f = open(txtName, 'w')
        f.write('Ref\n')
        # plotImg(tmp_img)
        res = processImgtoText(tmp_img)
        for j in range(len(res)):
            f.write(res[j][0:-2]+"\n")
        f.close()
        idxImagesToPrint = idxImagesToPrint + 1


def unskew_the_image(img):
    ThreshValue = 150
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
    if angle == 90:
        angle = 0
        plotImg(thresh, "angle à 90 deg")
        box = cv2.boxPoints(theRect)
        box = np.int0(box)
        cv2.drawContours(thresh, [box], 0, (255, 255, 255), 1)
        plotImg(thresh)
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


def decideWhatToDoWithTheResults(txt_value_from_complete_ref_box, txt_value_single_letter_box, txt_value_from_all_ref_boxes_from_the_selected_sheet, img):
    #TODO - finish to do the logic with the interface

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Tkinter works only with RGB and not BGR

    root = Tk()
    root.geometry("700x350")
    root.title("Could user please confirm or change reference ?")

    frame = tkinter.Frame(root)

    #Display image
    imgtk = ImageTk.PhotoImage(master=root, image=Image.fromarray(img.copy()))
    Label(root, image=imgtk).pack()

    # Display Text with tick box : txt_value_from_complete_ref_box
    complete_ref_box = tkinter.StringVar()
    complete_ref_box_check = tkinter.Checkbutton(
        root,
        text='txt_value_from_complete_ref_box ='+txt_value_from_complete_ref_box,
        variable=complete_ref_box,
        command=lambda: print(complete_ref_box.get()))
    # complete_ref_box_check.grid(column=0, row=2, sticky=tkinter.W)
    complete_ref_box_check.pack()

    # Display Text with tick box : txt_value_single_letter_box
    single_letter_box = tkinter.StringVar()
    single_letter_box_check = tkinter.Checkbutton(
        root,
        text='txt_value_single_letter_box = '+txt_value_single_letter_box,
        variable=single_letter_box,
        command=lambda: print(single_letter_box.get()))
    # single_letter_box_check.grid(column=0, row=3, sticky=tkinter.W)
    single_letter_box_check.pack()

    # Display Text with tick box : txt_value_from_all_ref_boxes_from_the_selected_sheet
    ref_boxes_from_the_selected_sheet = tkinter.StringVar()
    ref_boxes_from_the_selected_sheet_check = tkinter.Checkbutton(
        root,
        text='txt_value_from_all_ref_boxes_from_the_selected_sheet = '+txt_value_from_all_ref_boxes_from_the_selected_sheet,
        variable=ref_boxes_from_the_selected_sheet,
        command=lambda: print(ref_boxes_from_the_selected_sheet.get()))
    # ref_boxes_from_the_selected_sheet_check.grid(column=0, row=4, sticky=tkinter.W)
    ref_boxes_from_the_selected_sheet_check.pack()

    # Enter txt if nothing is okay
    Label(root, text='Please enter reference if nothing is correct regarding the above image').pack()
    keyword = Entry(root, width=30)
    keyword.focus()
    # keyword.grid(column=1, row=5, sticky=tkinter.W)
    keyword.pack()

    # Click on Button to exit after choice has been made
    tkinter.Button(root,
                   text='Next',
                   command=root.quit
                   ).pack()

    res = root.mainloop()



    return ""


def processImgtoText(img):
    minimumSizeOfARef = 10  # A02B-XXXX- = 10 char (4+1+4+1)
    padding_size = 20 # 15 pixels around text

    img = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # test for better recognition of the reference
    plotImg(img, "the Blured Padded Image")

    # Separate all of the text boxes for better reading of the OCR || psm=11 not so bad || psm=12 perfect !
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 12')
    # uncomment for debugging
    n_boxes = len(d['level'])
    theImage = img.copy()
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(theImage, (x, y), (x + w, y + h), (0, 0, 0), 1)
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

            # Unskew the text
            img_padded_and_unskewed = unskew_the_image(img_padded)
            # plotImg(img, "Unskewed")
            # From there image the text is straight

            #Blur the image with a Gaussian filter of 5x5
            img_unskewed_and_blured = cv2.GaussianBlur(img_padded_and_unskewed, (3, 3), 0)

            # In order to get another result for comparing
            txt_value_single_letter_box = ""
            # psm=10 treat image as a single char
            boxes = pytesseract.image_to_boxes(img_unskewed_and_blured, config='--psm 10 --oem 1 -c tessedit_char_whitelist=#-ABEFHJRSMTKVNG0123456789')
            for b in boxes.splitlines():
                b = b.split(' ')
                txt_value_single_letter_box = txt_value_single_letter_box + b[0]

            # plotImg(tmp_img, "tmp_img")
            txt_to_append = ""
            # psm=13 ng | psm=8 ng | psm=7 good | psm=6 ng | psm=4 ng
            txt_value_from_complete_ref_box = pytesseract.image_to_string(img_unskewed_and_blured, config='--psm 7 --oem 1 -c tessedit_char_whitelist=#-ABEFHJRSMTKVNG0123456789')
            txt_value_from_complete_ref_box = txt_value_from_complete_ref_box.rstrip()
            if txt_value_from_complete_ref_box != txt_value_single_letter_box \
                or txt_value_single_letter_box != txt_value_from_all_ref_boxes_from_the_selected_sheet \
                or txt_value_from_all_ref_boxes_from_the_selected_sheet != txt_value_from_complete_ref_box:
                print("Différent ! : " + txt_value_from_complete_ref_box.rstrip() + " || " + txt_value_single_letter_box + " || " + txt_value_from_all_ref_boxes_from_the_selected_sheet)
                txt_to_append = decideWhatToDoWithTheResults(txt_value_from_complete_ref_box,
                                                             txt_value_single_letter_box,
                                                             txt_value_from_all_ref_boxes_from_the_selected_sheet,
                                                             img_bordered)
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
    pages = convert_from_path(getPdfFile(), dpi=300)
    init_var()
    for i in range(len(pages)):
        images = select_boxes_in_the_png_file(pages[i])
        decideWhatToDoWithTheResults("A02B-0327-H010", "AQ2B-0327-H010", "A02B-0327~H010", images[0])
        # Récupérer le pointeur de la souris et trouver le contour le plus près suite à click sur l'image ?
        # select_box_by_click(img, trueContours)

        # imagesToPrint containt the rectangle for image analysis
        # Do also pytesseract
        mkdir_and_imwrite_selected_images_and_do_pytesseract(images)

        imagesToPrint.clear()

    extract_reference_and_create_excel_file()
