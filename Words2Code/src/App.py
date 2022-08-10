import subprocess
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageGrab


class App:

    def __init__(self):
        self.master = Tk()
        # sv_ttk.set_theme("dark")  # Set sv_ttk theme
        # Import the tcl file
        self.master.tk.call('source', 'themes/forest-dark.tcl')
        # Set the theme with the theme_use method
        ttk.Style().theme_use('forest-dark')
        self.master.title("Words2Code")
        self.outputBar = None
        self.codeView = None
        self.pattern = None
        self.Codetab = None
        self.HandWritetab = None
        self.tabControl = None
        self.c = None
        self.old_x = None
        self.old_y = None
        self.penwidth = 1
        self.drawWidgets()
        self.draw_canvas()

    def paint(self, e):
        self.pattern = self.c.create_line((self.old_x, self.old_y, e.x, e.y), width=self.penwidth, fill="black",
                                          capstyle=ROUND, smooth=True, tags="user_paint")
        self.old_x = e.x
        self.old_y = e.y

    def get_x_and_y(self, event):
        self.old_x, self.old_y = event.x, event.y

    def changeW(self, e):  # change Width of pen through slider
        self.penwidth = e

    def drawWidgets(self):
        self.tabControl = ttk.Notebook(self.master)
        self.tabControl.pack(expand=True, fill="both")

        self.Codetab = ttk.Frame(self.tabControl, width=500, height=500)
        self.HandWritetab = ttk.Frame(self.tabControl, width=500, height=500)

        self.Codetab.rowconfigure(0, weight=1)
        self.Codetab.columnconfigure(0, weight=1)

        self.HandWritetab.rowconfigure(0, weight=1)
        self.HandWritetab.columnconfigure(0, weight=1)

        self.Codetab.pack(fill='both', expand=True)
        self.HandWritetab.pack(fill='both', expand=True)

        self.tabControl.add(self.Codetab, text='Code')
        self.tabControl.add(self.HandWritetab, text='Pseudo-Editior')

        # CODE EDIT TAB

        runButt = ttk.Button(self.Codetab, width=5, style="Accent.TButton", text="Run", command=self.run)
        runButt.grid(row=0, column=3)

        uploadButt = ttk.Button(self.Codetab, text="Upload", width=6, style="Accent.TButton",
                                command=self.uploadPicture)
        uploadButt.grid(row=1, column=3, pady=150)

        clearButt = ttk.Button(self.Codetab, text="Clear", style="Accent.TButton", command=self.codeClear)
        clearButt.grid(row=2, column=3)

        lineNumbers = Label(self.Codetab, width=3, bg="#313131", fg="#cccaca",
                            font=('candara', 13))
        line = ""
        for i in range(1, 36):
            line += str(i) + "\n"
        line = line[:-1]
        lineNumbers.configure(text=line)
        lineNumbers.grid(row=0, column=0, rowspan=3, sticky="w")

        self.codeView = Text(self.Codetab, bg="white", fg="black", insertbackground="black",
                             font=('candara', 13, 'bold'))
        self.codeView.grid(row=0, column=2, rowspan=3, sticky="nswe")
        self.codeView.event_add('<<Paste>>', '<Control-v>')
        self.codeView.event_add('<<Copy>>', '<Control-c>')

        # Create a scrollbar
        scroll_bar = ttk.Scrollbar(self.Codetab, command=self.codeView.yview)
        self.codeView['yscrollcommand'] = scroll_bar.set
        scroll_bar.grid(row=0, column=1, rowspan=3, sticky="ns")

        self.outputBar = Label(self.Codetab, text="Ready", font=('calibri', 11), relief=GROOVE, anchor="w")
        self.outputBar.grid(row=3, column=0, columnspan=4, sticky="nswe")

        # PSEUDO CANVAS TAB

        pen_label = Label(self.HandWritetab, text='Pen Width', font=('candara', 11, 'bold'))
        pen_label.grid(row=0, column=1, sticky=N, pady=80)

        slider = ttk.Scale(self.HandWritetab, from_=1, to=4, command=self.changeW, orient=VERTICAL)
        slider.set(self.penwidth)
        slider.grid(row=0, column=1, sticky=N, pady=120)

        pseudo_clear = ttk.Button(self.HandWritetab, text="Clear", width=5, style="TButton",
                                  command=self.PseudoClear)
        pseudo_clear.grid(row=2, column=1, pady=80)

        convertButt = ttk.Button(self.HandWritetab, text="Convert", style="Accent.TButton", command=self.convert)
        convertButt.grid(row=2, column=1, sticky=S)

    def draw_canvas(self):
        self.c = Canvas(self.HandWritetab, bg="white")
        self.c.grid(row=0, column=0, rowspan=3, sticky="nswe")

        image = Image.open("images/Untitled-lines.png")
        resized_image = image.resize((770, 900), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(resized_image)

        self.c.bind('<B1-Motion>', self.paint)  # drawing the line
        self.c.bind("<Button-1>", self.get_x_and_y)

        self.c.create_image(0, 0, image=image, anchor='nw')
        self.master.mainloop()

    def run(self):
        file_name = "temp.py"
        with open(file_name, "w") as new_file:
            new_file.write(self.codeView.get(1.0, END))
        command = file_name  # the shell command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # Launch the shell command:
        output, err = process.communicate()
        if err:
            text = err
        else:
            byteData = output[:-2]
            text = byteData.decode('UTF-8')
            text = text.replace("\n", " ")
        if text == "":
            text = "Ready"
        self.outputBar.configure(text=text)

    def codeClear(self):
        self.codeView.delete(1.0, END)
        self.outputBar.configure(text="Ready")

    def PseudoClear(self):
        self.c.delete("user_paint")
        self.pattern = None

    def uploadPicture(self):
        filename = filedialog.askopenfilename(initialdir="/src",
                                              title="Select a File",
                                              filetypes=(("png files",
                                                          "*.png*"), ("all files",
                                                                      "*.*")))
        if filename != "":
            # Change label contents
            self.outputBar.configure(text="Opened: " + filename)
            # OCR FUNCTION HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def convert(self):
        if self.pattern is not None:
            x = self.HandWritetab.winfo_rootx() + self.c.winfo_x()
            y = self.HandWritetab.winfo_rooty() + self.c.winfo_y()
            x1 = x + self.c.winfo_width()
            y1 = y + self.c.winfo_height()
            ImageGrab.grab().crop((x, y, x1, y1)).save("resources/in/canvas.jpg")
            # OCR FUNCTION HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.c.delete("user_paint")
            self.tabControl.select(self.Codetab)


if __name__ == '__main__':
    App()
