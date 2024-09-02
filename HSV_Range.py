import cv2
import numpy as np
import argparse
import threading
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Configurar ArgumentParser para recibir el índice de la cámara
parser = argparse.ArgumentParser()
parser.add_argument("index_camera", help="Índice de la cámara para leer", type=int)
args = parser.parse_args()

# Crear objeto VideoCapture para capturar desde la cámara
capture = cv2.VideoCapture(args.index_camera)

# Verificar si la cámara se abrió correctamente
if not capture.isOpened():
    print("Error al abrir la cámara")
    exit()

# Configuración inicial de rango HSV
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Bloqueo para asegurar acceso seguro a las variables globales
lock = threading.Lock()

# Crear una figura para los sliders
fig, ax = plt.subplots(figsize=(8, 2))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8)
ax.axis('off')

# Crear ejes para los sliders
ax_hmin = plt.axes([0.1, 0.65, 0.35, 0.05])
ax_smin = plt.axes([0.1, 0.55, 0.35, 0.05])
ax_vmin = plt.axes([0.1, 0.45, 0.35, 0.05])

ax_hmax = plt.axes([0.60, 0.65, 0.35, 0.05])
ax_smax = plt.axes([0.60, 0.55, 0.35, 0.05])
ax_vmax = plt.axes([0.60, 0.45, 0.35, 0.05])

# Crear sliders para cada eje con valores iniciales
slider_hmin = Slider(ax_hmin, 'H mínimo', 0, 180, valinit=100)
slider_smin = Slider(ax_smin, 'S mínimo', 0, 255, valinit=100)
slider_vmin = Slider(ax_vmin, 'V mínimo', 0, 255, valinit=100)

slider_hmax = Slider(ax_hmax, 'H máximo', 0, 180, valinit=130)
slider_smax = Slider(ax_smax, 'S máximo', 0, 255, valinit=255)
slider_vmax = Slider(ax_vmax, 'V máximo', 0, 255, valinit=255)

# Función para actualizar la máscara según los sliders
def update(val):
    global lower_blue, upper_blue
    with lock:  # Usar el bloqueo para asegurar que no haya acceso simultáneo conflictivo
        h_min = slider_hmin.val
        s_min = slider_smin.val
        v_min = slider_vmin.val
        h_max = slider_hmax.val
        s_max = slider_smax.val
        v_max = slider_vmax.val
        lower_blue = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_blue = np.array([h_max, s_max, v_max], dtype=np.uint8)

# Vincular sliders con la función de actualización
slider_hmin.on_changed(update)
slider_smin.on_changed(update)
slider_vmin.on_changed(update)
slider_hmax.on_changed(update)
slider_smax.on_changed(update)
slider_vmax.on_changed(update)

# Crear ejes para botón de reinicio y crear el botón
resetax = plt.axes([0.4, 0.2, 0.2, 0.1])
button = Button(resetax, 'Reset', color='gold', hovercolor='skyblue')

# Función para restablecer los sliders a los valores iniciales
def resetSlider(event):
    slider_hmin.reset()
    slider_smin.reset()
    slider_vmin.reset()
    slider_hmax.reset()
    slider_smax.reset()
    slider_vmax.reset()

# Vincular botón de reinicio con la función de restablecimiento
button.on_clicked(resetSlider)

# Función para mostrar la captura de video con detección de color
def show_video():
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        # Convertir el frame a HSV y aplicar la máscara de color
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with lock:  # Asegurar que no se cambien los valores mientras se leen
            mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        frame = cv2.resize(frame, (0,0), fx=0.7, fy=0.7)
        res = cv2.resize(res, (0,0), fx=0.7, fy=0.7)
        # Mostrar los frames
        cv2.imshow('Input frame from the camera', frame)
        cv2.imshow('Masked output (color detection)', res)

        # Salir si se presiona 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Liberar recursos
    capture.release()
    cv2.destroyAllWindows()

# Ejecutar la captura de video en un hilo separado
video_thread = threading.Thread(target=show_video)
video_thread.start()

# Mostrar la ventana de sliders
plt.show()

# Esperar a que el hilo de video termine
video_thread.join()
