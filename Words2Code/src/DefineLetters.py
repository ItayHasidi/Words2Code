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
