import os
import subprocess
import sys, string
import time

from src import OCR


class DefineLetters:
    # for filename in os.listdir(os.getcwd()):
    #     print(filename)
    @staticmethod
    def callExtract():
        subprocess.call('ExtractLetters.exe')
        time.sleep(1)
        OCR.run_example()
# os.system('start src\\ExtractLetters.exe') 
# o = subprocess.Popen(['cmd', '/c', 'src\\ExtractLetters.exe'])
# o.wait()
# proc = subprocess.Popen(["../src", "saved-canvas.jpg"]) # saved-canvas.jpg  ExtractLetters.exe
# proc.wait()
