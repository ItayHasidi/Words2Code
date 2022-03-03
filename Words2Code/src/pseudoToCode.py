# this function opens the output file created from the ocr algo
def extract_txt():
    file = open("../output/output_java_3.txt", "r")
    return file


def check_syntax():
    file = extract_txt()
    lines = file.readlines()
    for line in lines:
        # print(line)
        words = line.split(" ")

        print(words)

check_syntax()
# print(extract_txt().read())