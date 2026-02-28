import cv2
import numpy as np

# ---------- Funciones auxiliares ----------
def adjust_gamma(image, gamma=1.25):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

def unsharp_mask(img, amount=0.7, radius=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def enhance_face(roi):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    out = adjust_gamma(out, gamma=1.25)
    out = cv2.fastNlMeansDenoisingColored(out, None, 6, 6, 7, 21)
    out = unsharp_mask(out)

    return out

# ---------- Rutas ----------
input_path = r"C:\Users\User\Documents\revisar.jpeg"
output_path = r"C:\Users\User\Documents\revisar_mejorada.jpg"

# ---------- Cargar imagen ----------
img = cv2.imread(input_path)
if img is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

# Pre-ajuste leve para ayudar a detectar rostros
img = adjust_gamma(img, gamma=1.15)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- Detector de rostros ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(50, 50)
)

result = img.copy()

# ---------- Procesar cada rostro ----------
for (x, y, w, h) in faces:
    pad = int(0.18 * w)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

    roi = result[y1:y2, x1:x2]
    enhanced = enhance_face(roi)

    blended = cv2.addWeighted(enhanced, 0.85, roi, 0.15, 0)
    result[y1:y2, x1:x2] = blended

# ---------- Guardar resultado ----------
cv2.imwrite(output_path, result)

print(f"✔ Imagen mejorada guardada en:\n{output_path}")
print(f"✔ Rostros detectados: {len(faces)}")
