# Check QRs

> This script is part of the article [How to create QR codes with Stable Diffusion](https://www.felixsanz.dev/how-to-create-qr-codes-with-stable-diffusion) ([felixsanz.dev](https://www.felixsanz.dev))

This Python script checks the validity of QR codes. It uses 3 different QR code reading libraries to **filter the codes that are valid in all libraries at the same time**.

What it does:

1. It is in charge of reading the `.png` images that are in the `./images` folder.
2. Check the validity of the QR code using the `opencv-python`, `pyzbar` and `zxing` libraries. You must pass as the first (and only) argument the text or URL with which you created the QR code so that it can be verified that the reading result is the same.
3. If the QR code is valid in all 3 implementations, the script copies the image to the `./valid` folder.

## Installation

Clone this repository and open a terminal in the location. Then create a Python virtual environment and activate it.

```bash
git clone https://github.com/felixsanz/dev
cd articles/how-to-create-qr-codes-with-stable-diffusion

python -m venv .venv
source .venv/bin/activate
```

Then install dependencies.

```bash
pip install -r requirements.txt
```

## Usage

Copy the `.png` images of the QR codes you want to check into the `./images` folder.

Run the script and see the results.

```bash
python check.py "https://www.felixsanz.dev"
```

```
[cv2 pyzbar zxing] ./images/00218-2724904587.png
[cv2 pyzbar zxing] ./images/00235-4081443668.png
[cv2 pyzbar zxing] ./images/00159-2755401775.png
```

The names of the libraries appear in green or red depending on the result of the library. When an image shows all 3 libraries in green, it will **be copied** to the `./valid` folder.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
