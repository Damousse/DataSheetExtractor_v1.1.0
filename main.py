# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# DBM - Command line to create .exe file : pyinstaller .\main.py inside the venv
# DBM - Tesseract v5.0 installed on Windows
# DBM - 200 dpi seems optimum & letters might have a font size of 30-33 (pixels) regarding the litterature
# DBM - Python v=3.7 (venv)
# https://groups.google.com/g/tesseract-ocr/c/Wdh_JJwnw94/m/24JHDYQbBQAJ

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
from tkinter import ttk
import shutil
import regex
import math
try:
    from PIL import Image, ImageTk
except ImportError:
    import Image, ImageTk

root = Tk()
root.withdraw()
selected_text = ""
TextTab = []
ImageforROI = []
drawing_state = 0
p0 = (0,0)
p1 = (0,0)
p2 = (0,0)
p3 = (0,0)
points = []
regions = []
VariableTextSelected = tkinter.IntVar()
UserInputText = tkinter.StringVar()


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


def selectROI_old(tmp):
    fromCenter = False
    showCrosshair = True
    regions = []
    h, w = tmp.shape[:2]
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


def selectROI(tmp):
    global ImageforROI
    title = 'ROI Selector'
    h, w = tmp.shape[:2]
    ratio = int(h / 900)
    ImageforROI = cv2.resize(tmp, (int(w / ratio), int(h / ratio)))
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)
    while True:
        cv2.imshow(title, ImageforROI)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    return regions


def mouse(event, x, y, flags, param):
    global p0, p1, p2, p3, drawing_state, ImageforROI, points, regions

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_state == 0:
            p0 = [x, y]
            drawing_state = 1
        elif drawing_state == 1:
            p1 = [x, y]
            cv2.line(ImageforROI, p0, p1, (0, 255, 0), 2)
            points.append(p0)
            drawing_state = 2
        elif drawing_state == 2:
            p2 = [x, y]
            cv2.line(ImageforROI, p1, p2, (0, 255, 0), 2)
            points.append(p1)
            drawing_state = 3
        elif drawing_state == 3:
            p3 = [x, y]
            cv2.line(ImageforROI, p2, p3, (0, 255, 0), 2)
            points.append(p2)
            drawing_state = 4

    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()
        drawing_state = 0

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_state == 4:
            cv2.line(ImageforROI, p3, p0, (0, 255, 0), 2)
            points.append(p3)
            drawing_state = 0
            regions = constructRegionsFromPoints(points)


def constructRegionsFromPoints(points):
    points = np.array(points)
    #y_axis_sorted_box = points[points[:, 1].argsort()]
    #tmp_top = np.array([y_axis_sorted_box[0], y_axis_sorted_box[1]])
    #tmp_bottom = np.array([y_axis_sorted_box[-2], y_axis_sorted_box[-1]])
    #(top_left, top_right) = tmp_top[tmp_top[:, 0].argsort()]
    #(bottom_left, bottom_right) = tmp_bottom[tmp_bottom[:, 0].argsort()]
    # TODO : extract the correct region from original image with the polygon shape from points
    # TODO : Maybe do a mask over ImageForROI in order to extract zone then compute the minrectArea
    mask = np.zeros(ImageforROI.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, pts=[points], color=(255, 255, 255))
    masked = cv2.bitwise_and(ImageforROI, ImageforROI, mask=mask)
    plotImg(masked, "Masked Image")
    # need to extract the region from original image and then mask the imahe with a white rectangular background
    return []


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
            f.write(res[j]+"\n")
        f.close()
        idxImagesToPrint = idxImagesToPrint + 1


def computeAngle(box):
    y_axis_sorted_box = box[box[:, 1].argsort()]
    # tmp_top = np.array([y_axis_sorted_box[0], y_axis_sorted_box[1]])
    tmp_bottom = np.array([y_axis_sorted_box[-2], y_axis_sorted_box[-1]])
    # (top_left, top_right) = tmp_top[tmp_top[:, 0].argsort()]
    (bottom_left, bottom_right) = tmp_bottom[tmp_bottom[:, 0].argsort()]

    angle = math.atan(math.fabs(bottom_left[1]-bottom_right[1])/math.fabs(bottom_left[0]-bottom_right[0]))*180/math.pi
    if bottom_left[1] > bottom_right[1]:# pente positive
        angle = - angle

    return angle


def unskew_the_image(img):
    ThreshValue = 150
    img2 = cv2.bitwise_not(img)
    # plotImg(img2, "img2")
    # from here image is black in background and light for the text
    thresh = cv2.threshold(img2, ThreshValue, 255, cv2.THRESH_BINARY)[1]
    # plotImg(thresh, "Thresh")
    val = np.where(thresh > 0)
    coords = np.column_stack([val[1], val[0]])
    theRect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(theRect)
    box = np.int0(box)
    # cv2.drawContours(thresh, [box], 0, (255, 255, 255), 1)
    # plotImg(thresh)
    angle = computeAngle(box)
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    #rotate the image of "angle" in counterclockwise way
    M = cv2.getRotationMatrix2D(center, angle, 1)
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
    # plotImg(thresh)


def exitTkinterWindow(root, keyword):
    global selected_text, VariableTextSelected, TextTab
    TextTab.append(keyword.get())
    selected_text = TextTab[VariableTextSelected.get()]
    root.quit()


def decideWhatToDoWithTheResults(txt_value_from_complete_ref_box, txt_value_single_letter_box, txt_value_from_all_ref_boxes_from_the_selected_sheet, img):
    global selected_text, VariableTextSelected, root

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Tkinter works only with RGB and not BGR
    TextTab.append(txt_value_from_complete_ref_box)
    TextTab.append(txt_value_single_letter_box)
    TextTab.append(txt_value_from_all_ref_boxes_from_the_selected_sheet)
    scoreTab = computeConfidenceScore(TextTab)

    root.deiconify()

    root.geometry("800x600")
    root.title("Could user please confirm or change reference ?")

    # Display image
    imgtk = ImageTk.PhotoImage(master=root, image=Image.fromarray(img.copy()))
    lbl = Label(root, image=imgtk).pack(padx=20, pady=20)

    # Display Text with tick box : txt_value_from_complete_ref_box
    complete_ref_box_check = tkinter.Radiobutton(
        root,
        text=txt_value_from_complete_ref_box + " (match approx." + scoreTab[0] + "%)",
        variable=VariableTextSelected,
        value=0)
    complete_ref_box_check.config(font=("MS Sans Serif", 12))
    complete_ref_box_check.pack(padx=20, pady=20)

    # Display Text with tick box : txt_value_single_letter_box
    single_letter_box_check = tkinter.Radiobutton(
        root,
        text=txt_value_single_letter_box + " (match approx." + scoreTab[1] + "%)",
        variable=VariableTextSelected,
        value=1)
    single_letter_box_check.config(font=("MS Sans Serif", 12))
    single_letter_box_check.pack(padx=20, pady=20)

    # Display Text with tick box : txt_value_from_all_ref_boxes_from_the_selected_sheet
    ref_boxes_from_the_selected_sheet_check = tkinter.Radiobutton(
        root,
        text=txt_value_from_all_ref_boxes_from_the_selected_sheet + " (match approx." + scoreTab[2] + "%)",
        variable=VariableTextSelected,
        value=2)
    ref_boxes_from_the_selected_sheet_check.config(font=("MS Sans Serif", 12))
    ref_boxes_from_the_selected_sheet_check.pack(padx=20, pady=20)

    # Display Text with tick box : UserInputText
    UserInputText = tkinter.Radiobutton(
        root,
        text='UserInputText = ',
        variable=VariableTextSelected,
        value=3)
    UserInputText.pack(pady=(20, 0))
    UserInputText.config(font=("MS Sans Serif", 12))

    # Enter txt if nothing is okay
    keyword = Entry(root, width=20, font=("MS Sans Serif", 12), justify='center')
    keyword.insert(0, txt_value_from_complete_ref_box)
    keyword.pack(pady=(0, 20))

    #Style of the button
    style = ttk.Style()
    style.configure('W.TButton', font=('calibri', 10, 'bold', 'underline'), justify='center')

    # Click on Button to exit after choice has been made
    btn = ttk.Button(root,
               text='Next',
               style='W.TButton',
               command=lambda: exitTkinterWindow(root, keyword)
               ).pack(pady=40, padx=20, ipadx=18, ipady=30)

    root.mainloop()

    # Start again with fresh ref
    TextTab.clear()
    for widget in root.winfo_children():
       widget.destroy()
    root.withdraw()

    return selected_text


def computeConfidenceScore(TextTab):
    refNum_Regex = regex.compile('(^A0[0-9]B[-]0[0-9][0-9][0-9][-][A-Z][0-9][0-9][0-9]){i,s,d}')
    refWithDiese4_Regex = regex.compile('(^A0[0-9]B[-]0[0-9][0-9][0-9][-][A-Z][0-9][0-9][0-9][#]....){i,s,d}')
    refWithDiese3_Regex = regex.compile('(^A0[0-9]B[-]0[0-9][0-9][0-9][-][A-Z][0-9][0-9][0-9][#]...){i,s,d}')
    refWithDiese2_Regex = regex.compile('(^A0[0-9]B[-]0[0-9][0-9][0-9][-][A-Z][0-9][0-9][0-9][#]..){i,s,d}')
    refWithDiese1_Regex = regex.compile('(^A0[0-9]B[-]0[0-9][0-9][0-9][-][A-Z][0-9][0-9][0-9][#].){i,s,d}')
    confidenceScore = []
    for txt in TextTab:
        if txt != "":
            (i1, s1, d1) = regex.fullmatch(refWithDiese1_Regex, txt).fuzzy_counts
            (i2, s2, d2) = regex.fullmatch(refWithDiese2_Regex, txt).fuzzy_counts
            (i3, s3, d3) = regex.fullmatch(refWithDiese3_Regex, txt).fuzzy_counts
            (i4, s4, d4) = regex.fullmatch(refWithDiese4_Regex, txt).fuzzy_counts
            (i5, s5, d5) = regex.fullmatch(refNum_Regex, txt).fuzzy_counts
            best_match = min(min(min(min(i1+s1+d1,i2+s2+d2), i3+s3+d3), s4+i4+d4), s5+i5+d5)
            if best_match < 0:
                best_match = 0
            elif best_match > 100:
                best_match = 100
            confidenceScore.append(str(int(100*(1-best_match/len(txt)))))
        else:
            confidenceScore.append("NaN")
    return confidenceScore


def processImgtoText(myImg):
    minimumSizeOfARef = 10  # A02B-XXXX- = 10 char (4+1+4+1)
    padding_size = 20 # 15 pixels around text

    myImg = cv2.copyMakeBorder(myImg, padding_size, padding_size, padding_size, padding_size,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])
    myImg = cv2.GaussianBlur(myImg, (3, 3), 0)

    myImg = unskew_the_image(myImg)

    # plotImg(myImg, "the Blured Padded Unskewed Image")

    # Separate all of the text boxes for better reading of the OCR || psm=11 not so bad || psm=12 perfect !
    d = pytesseract.image_to_data(myImg, output_type=Output.DICT, config='--psm 12')
    # uncomment for debugging
    # n_boxes = len(d['level'])
    # theImage = myImg.copy()
    # for i in range(n_boxes):
        # (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # cv2.rectangle(theImage, (x, y), (x + w, y + h), (0, 0, 0), 1)
    # plotImg(theImage, "the Image with Boxes")

    n_boxes = len(d['level'])
    res = []
    txt_value_from_all_ref_boxes_from_the_selected_sheet = ""
    for i in range(n_boxes):
        # centers = []
        if d["text"][i] != "" and len(d["text"][i]) > minimumSizeOfARef:
            txt_value_from_all_ref_boxes_from_the_selected_sheet = d["text"][i]
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img_bordered = myImg[y:y + h, x:x + w]

            # Padding of the image for better results of the OCR
            img_padded = cv2.copyMakeBorder(img_bordered, padding_size, padding_size, padding_size, padding_size,
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])

            # plotImg(img_padded)

            # Unskew the text
            img_padded_and_unskewed = unskew_the_image(img_padded)
            # plotImg(img_padded_and_unskewed, "Unskewed")
            # From there image the text is straight

            #Blur the image with a Gaussian filter of 5x5
            img_unskewed_and_blured = cv2.GaussianBlur(img_padded_and_unskewed, (3, 3), 0)
            # plotImg(img_unskewed_and_blured, "img_unskewed_and_blured")
            # In order to get another result for comparing
            txt_value_single_letter_box = ""

            # psm=10 treat image as a single char
            # boxes = pytesseract.image_to_boxes(img_unskewed_and_blured, config='--psm 10 --oem 1 -c tessedit_char_whitelist=#-ABEFHJRSMTKVNG0123456789')
            boxes = pytesseract.image_to_boxes(img_unskewed_and_blured)
            theImage2 = img_unskewed_and_blured.copy()
            theImage2 = cv2.adaptiveThreshold(theImage2, 170, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
            height, width = theImage2.shape  # assumes color image
            for b in boxes.splitlines():
                b = b.split(' ')
                theImage2 = cv2.rectangle(theImage2, (int(b[1]), height - int(b[2])), (int(b[3]), height - int(b[4])), (0, 255, 0), 1)
                txt_value_single_letter_box = txt_value_single_letter_box + b[0]
            plotImg(theImage2, "the Image with Letter Boxes")

            txt_to_append = ""
            # psm=13 ng | psm=8 ng | psm=7 good | psm=6 ng | psm=4 ng
            txt_value_from_complete_ref_box = pytesseract.image_to_string(img_unskewed_and_blured, config='--psm 7 --oem 1 -c tessedit_char_whitelist=#-ABEFHJRSMTKVNG0123456789')
            txt_value_from_complete_ref_box = txt_value_from_complete_ref_box.rstrip()
            if txt_value_from_complete_ref_box != txt_value_single_letter_box \
                or txt_value_single_letter_box != txt_value_from_all_ref_boxes_from_the_selected_sheet:
                print("Différent ! : " + txt_value_from_complete_ref_box + " || " + txt_value_single_letter_box + " || " + txt_value_from_all_ref_boxes_from_the_selected_sheet)
                txt_to_append = decideWhatToDoWithTheResults(txt_value_from_complete_ref_box,
                                                             txt_value_single_letter_box,
                                                             txt_value_from_all_ref_boxes_from_the_selected_sheet,
                                                             img_padded_and_unskewed)
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
        # pdf = pytesseract.image_to_pdf_or_hocr(pages[i], extension='pdf')
        # with open('test.pdf', 'w+b') as f:
            # f.write(pdf)  # pdf type is bytes by default
        images = select_boxes_in_the_png_file(pages[i])

        # imagesToPrint containt the rectangle for image analysis and do also pytesseract
        mkdir_and_imwrite_selected_images_and_do_pytesseract(images)

        imagesToPrint.clear()

    extract_reference_and_create_excel_file()
