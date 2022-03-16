from autocorrect import Speller


def check_syntax(file_name):
    # file = open(file_name, "r")
    # lines = file.readlines()
    # for line in lines:
    # words = line.split(" ")
    # spell = Speller(only_replacements=True)
    spell = Speller()
    print(spell(file_name))


if __name__ == '__main__':
    check_syntax()
# print(extract_txt().read())
