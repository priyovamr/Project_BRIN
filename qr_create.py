import qrcode
import pandas as pd

def create_qr_code(text, file_name):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    path = f"C:/Users/User/OneDrive - mail.ugm.ac.id/Dokumen/Kuliah/Asdos/Project_BRIN/{file_name}"
    img.save(path)

if __name__ == "__main__":
    content = "PERANCANGAN LOW-COST TESTBED UNTUK VALIDASI LOKASI DAN ORIENTASI MOBILE ROBOT"
    nama = "QR.png"
    create_qr_code(content,nama)