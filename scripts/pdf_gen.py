from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image, PageBreak,  Spacer, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm, inch
from reportlab.platypus.flowables import KeepInFrame
import matplotlib.pyplot as plt
import io 
import pandas as pd

class pdf:
    # Class for pdf generate
    # Variables:
    # styles - styles for generated pdf
    # elements - content of generated pdf
    # doc - document object for pdf
    styles = None
    elements = None
    doc = None
    pagesize = None
    
    def __init__(self, output_filename: str, header_text : str = None):
        # Function for initiation of the class
        # Parameters:
        # output_filename: str - name of the generated pdf
        self.pagesize = (11.5*inch, 11.5*inch)
        self.doc = SimpleDocTemplate(output_filename, pagesize=self.pagesize)
        self.styles = getSampleStyleSheet()
        self.elements = []
        font_size = 30
        font_name = 'Helvetica-Bold'
        if header_text != None:
            style = ParagraphStyle(
            name='HeaderStyle',
            fontName=font_name,
            fontSize=font_size,
            textColor=colors.black,
            alignment=1,
            spaceAfter=80)
            header_paragraph = Paragraph(header_text, style)
            self.elements.append(header_paragraph)
            self.elements.append(Paragraph("<br/>", self.styles['Normal']))  

        

    def add_title(self, title : str):
        # Function for adding the title
        # Parameters:
        # title : str - title to add to pdf
        title_style = self.styles['Heading1']
        title_paragraph = Paragraph(title, title_style)
        self.elements.append(title_paragraph)
        self.elements.append(Paragraph("<br/>", self.styles['Normal']))  

    def add_table(self, table_data: list):
        # Adding table data to pdf
        # Parameters:
        # table_data:  list of lists - columns names + row by row information about table 
          # Calculate the number of columns and rows in the table
        num_columns = len(table_data[0])
        num_rows = len(table_data)
        page_width, page_height = self.pagesize
        page_width *= 0.8
        column_width = page_width / num_columns
        table = Table(table_data, colWidths=[column_width] * num_columns)
        style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header row background color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header row text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center alignment for all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold font for header row
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Bottom padding for header row
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Background color for data rows
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Grid lines for all cells
        ('WORDWRAP', (0, 0), (-1, -1), 'CHAR'),  # Word wrap cell content
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Vertical alignment for all cells (MIDDLE for word wrapping)
        ('SIZE', (0, 0), (-1, -1), 10),  # Font size for all cells
    ])
        table.setStyle(style)
        table_in_frame = KeepInFrame(column_width * num_columns, page_height, [table])
        self.elements.append(table_in_frame)
        
    def add_dataframe(self, df_data: pd.DataFrame):
        # Adding datafrmae data to pdf
        # Parameters:
        # df_data: pandas DataFrame -- dataframe to add
        df_data = df_data.round(2)
        df_table_data = [df_data.columns.tolist()] + df_data.values.tolist()
        rows_data = df_data.index.tolist()
        if len(rows_data):
            df_table_data = [[""] + df_table_data[0]] + [[rows_data[i]] +  df_table_data[i + 1] for i in range(len(rows_data))]
        self.add_table(df_table_data)

    def add_text(self, text: str):
         # Adding text output to the PDF document
         # Parameters:
         # text :  str -  text you want to add to PDF
         for line in text.split('\n'):
            if '\t' in line:
                self.elements.extend([Paragraph(part, self.styles['Normal']) for part in line.split('\t')])
            else:
                self.elements.append(Paragraph(line, self.styles['Normal']))
            self.elements.append(Spacer(1, 12))  


    def add_image(self, fig_data):
         # Adding matprolib figure to the PDF document
         # Parameters:
         # fig_data : bytes - figure data to add to PDF
        img_byte_arr = io.BytesIO()
        fig_data.savefig(img_byte_arr, format='PNG')
        img = Image(img_byte_arr)
        page_width, page_height = self.pagesize
        img_width, img_height = img.drawWidth, img.drawHeight
        scale_factor = page_width / img_width
        new_img_width, new_img_height = img_width * scale_factor, img_height * scale_factor
        img.drawWidth, img.drawHeight = new_img_width, new_img_height
        self.elements.append(img)

    def generate_pdf(self):
        # Function to generate PDF from elements 
        self.doc.build(self.elements)