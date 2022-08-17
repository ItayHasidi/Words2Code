import os


def main():
    file = open('../output/output_text.txt')
    file_new = open('../src/temp.py', 'w')
    line = file.readline()
    arr = []
    l = 0  # count how much labs threre is. /#

    while line:

        arr = line.split(" ")
        a = arr[0]
        # print(l)
        for i in range(l):
            a = "\t" + a
        arr[0] = a

        # print('++++++' ,'\n')

        n = len(arr)
        # print(arr)
        for i in range(n):

            a = str(arr[i])

            if a.endswith("++", 0, len(a) - 1):
                # print(a)
                a = a.replace("++", " += 1")
                # print(a)

            if a.endswith("--", 0, len(a) - 1):
                a = a.replace("--", " -= 1")

            if a == "to":
                a = a.replace("to", "in")
                arr[i] = a
            arr[i] = a
        # end for.

        if arr[0] == '\t\tfor' or arr[0] == '\tfor' or arr[0] == 'for' or arr[0] == '\t\tif' or arr[0] == '\tif' or arr[0] == 'if' or  arr[0] == '\t\twhile' or arr[0] == '\twhile' or arr[0] == 'while'  :
            # print("w")
            l += 1
            a = arr[n - 1]
            a = a.replace("\n", ":\n")
            arr[n - 1] = a

        if  arr[0] == '\t\tend-for\n' or arr[0] == '\tend-for\n' or arr[0] == '\tend-while\n' or arr[0] == '\tend-if\n' or arr[0] == 'end-if\n' or arr[0] == 'end-for\n' or arr[0] == 'end-while\n' or arr[0] == '\t\tend-while\n' or arr[0] == '\t\tend-if\n' :
            # print("what")
            l -= 1
            arr.pop()

        for i in range(len(arr)):
            a = str(arr[i]) + " "

            file_new.write(a)

        line = file.readline()

    file_new.close()


if __name__ == '__main__':
    main()
