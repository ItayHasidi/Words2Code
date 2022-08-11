import os
import subprocess
import sys, string


class DefineLetters:

    @staticmethod
    def callExtract():
        subprocess.call('ExtractLetters.exe')


# for filename in os.listdir(os.getcwd()):
#     print(filename)
# subprocess.call('ExtractLetters.exe')
# os.system('start src\\ExtractLetters.exe') 
# o = subprocess.Popen(['cmd', '/c', 'src\\ExtractLetters.exe'])
# o.wait()
# proc = subprocess.Popen(["../src", "saved-canvas.jpg"]) # saved-canvas.jpg  ExtractLetters.exe
# proc.wait()
