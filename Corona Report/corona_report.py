from fpdf import FPDF
from datetime import datetime, timedelta, date
from corona_plots import make_plot_images
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))


WIDTH = 210
HEIGHT = 297


class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 24)

        # Title
        self.set_x(0)
        self.cell(WIDTH, 25, "SCANDINAVIA", 1, 0, "C", True)
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font("Arial", "I", 8)
        # Page number
        self.cell(0, 10, str(self.page_no()) + " ({nb})", 0, 0, "C")


def create_title(day, pdf):
    # Unicode is not yet supported in the py3k version; use windows-1252 standard font
    # pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "", 24)
    pdf.ln(60)
    pdf.set_x(25)
    pdf.cell(WIDTH / 2, 20, "Covid Analytics Report", 0, 0, "L", False)
    # pdf.write(5, f"Covid Analytics Report")
    pdf.ln(15)
    pdf.set_font("Arial", "", 16)
    pdf.set_x(25)
    pdf.cell(WIDTH / 2, 8, f"{day}", 0, 0, "L", False)
    # pdf.write(4, f"{day}")
    pdf.ln(5)


def create_report(day=date.today(), filename="report.pdf"):
    pdf = PDF()
    pdf.alias_nb_pages()

    # Make plots and save images
    make_plot_images()

    # First page
    pdf.add_page()
    create_title(str(day), pdf)

    # Second page
    pdf.add_page()

    pdf.image("./tmp/CUMULATIVE_CASES.png", 25, 40, WIDTH - 50)
    pdf.image("./tmp/CUMULATIVE_DEATHS.png", 25, 160, WIDTH - 50)

    # Third page
    pdf.add_page()

    pdf.image("./tmp/DAILY_INCREASE.png", 25, 40, WIDTH - 50)
    pdf.image("./tmp/DEATHS_PER_DAY.png", 25, 160, WIDTH - 50)

    # Output report
    f = str(date.today()) + " " + filename
    pdf.output(f, "F")


if __name__ == "__main__":
    create_report()
