import os
import subprocess
import sys, string
import time
from src import OCR, Extract_Line, convertToPython


class DefineLetters:
    # for filename in os.listdir(os.getcwd()):
    #     print(filename)
    @staticmethod
    def callExtract():
        # subprocess.call('ExtractLetters.exe')
        # time.sleep(1)
        # OCR.run_example()

        Extract_Line.main()
        OCR.run_example()
        convertToPython.main()
