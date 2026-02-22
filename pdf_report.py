# pdf_report.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table
from reportlab.platypus import TableStyle

def generate_pdf(filename, metrics_dict):

    doc = SimpleDocTemplate(filename)
    elements = []

    style = ParagraphStyle(name='Normal', fontSize=14)
    elements.append(Paragraph("Stock Analytics Executive Report", style))
    elements.append(Spacer(1, 0.3*inch))

    data = [[k, str(v)] for k,v in metrics_dict.items()]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))

    elements.append(table)
    doc.build(elements)
