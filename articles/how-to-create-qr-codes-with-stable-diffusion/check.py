import os
import sys
import glob
import shutil
import cv2
from pyzbar import pyzbar
import zxing

if len(sys.argv) < 2 or not sys.argv[1]:
  print('Missing first argument (text or url)')
  print('Usage: python check.py \"https://www.felixsanz.dev"')
  sys.exit()

def read_qr_code_with_cv2(file):
  try:
    detector = cv2.QRCodeDetector()
    img = cv2.imread(file)

    value = detector.detectAndDecode(img)[0]

    return value
  except:
    return

def read_qr_code_with_pyzbar(file):
  try:
    img = cv2.imread(file)

    value = pyzbar.decode(img)[0].data.decode()

    return value 
  except:
    return

def read_qr_code_with_zxing(file):
  try:
    detector = zxing.BarCodeReader()
    value = detector.decode(file).parsed

    return value
  except:
    return

PASS = '\033[32m'
FAIL = '\033[31m'
RESET = '\033[0m'

text = sys.argv[1]

for file in glob.glob('./images/*.png'):
  cv2_code = read_qr_code_with_cv2(file)
  cv2_result = (cv2_code == text)
  cv2_color = PASS if cv2_result else FAIL

  pyzbar_code = read_qr_code_with_pyzbar(file)
  pyzbar_result = (pyzbar_code == text)
  pyzbar_color = PASS if pyzbar_result else FAIL

  zxing_code = read_qr_code_with_zxing(file)
  zxing_result = (zxing_code == text)
  zxing_color = PASS if zxing_result else FAIL

  print(f'[{cv2_color}cv2{RESET} {pyzbar_color}pyzbar{RESET} {zxing_color}zxing{RESET}] {file}')

  if (cv2_result and pyzbar_result and zxing_result):
    shutil.copy2(file, file.replace('/images/', '/valid/'))
