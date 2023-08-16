import fitz

pdffile = "file.pdf"
doc = fitz.open(pdffile)
page = doc.load_page(1)  # number of page

zoom = 10    # zoom factor
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)
output = "outfile.png"
pix.save(output)
doc.close()