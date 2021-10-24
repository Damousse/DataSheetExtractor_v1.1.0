# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pdf2image import convert_from_path

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    images = convert_from_path('example.pdf')

    for i, image in enumerate(images):
        fname = "image" + str(i) + ".png"
        image.save(fname, "PNG")
