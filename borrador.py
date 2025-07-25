import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

# Parámetros de las cajas
box_width = 2.5
box_height = 1.0
main_x = 0
side_x = 3.2   # x para cajas laterales (derecha)
y_start = 7
y_gap = 1.8

# Función para dibujar caja con texto
def draw_box(x, y, text):
    rect = patches.FancyBboxPatch((x - box_width/2, y - box_height/2),
                                  box_width, box_height,
                                  boxstyle="round,pad=0.3",
                                  edgecolor="black",
                                  facecolor="white")
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10)

# Cajas principales (columna central)
texts_main = [
    "Records identified\nthrough Scopus\n(n = 568)",
    "Records identified\nthrough Web of Science\n(n = 375)",
    "Records after duplicates removed\n(n = 681)",
    "Records screened\n(title/abstract)\n(n = 681)",
    "Full-text articles assessed\n(n = 32)",
    "Studies included in qualitative synthesis\n(n = 32)"
]

y_positions_main = [y_start, y_start - y_gap, y_start - 2*y_gap,
                    y_start - 3*y_gap, y_start - 4*y_gap,
                    y_start - 5*y_gap]

# Dibujo cajas principales
for text, y in zip(texts_main, y_positions_main):
    draw_box(main_x, y, text)

# Cajas laterales
side_boxes = {
    "Duplicates removed\n(n = 262)": y_start - 2*y_gap,
    "Records excluded\n(n = 649)": y_start - 3*y_gap
}

for text, y in side_boxes.items():
    draw_box(side_x, y, text)

# Dibujar flechas entre cajas principales
for y1, y2 in zip(y_positions_main[:-1], y_positions_main[2:]):  # desde Scopus/WoS hasta deduplicado
    ax.annotate("", xy=(main_x, y1 - 0.9), xytext=(main_x, y2 + 0.9),
                arrowprops=dict(arrowstyle="->"))

# Flecha de deduplicado a screening
ax.annotate("", xy=(main_x, y_positions_main[2] - 0.9),
            xytext=(main_x, y_positions_main[3] + 0.9),
            arrowprops=dict(arrowstyle="->"))

# Flecha de screening a full-text
ax.annotate("", xy=(main_x, y_positions_main[3] - 0.9),
            xytext=(main_x, y_positions_main[4] + 0.9),
            arrowprops=dict(arrowstyle="->"))

# Flecha de full-text a included
ax.annotate("", xy=(main_x, y_positions_main[4] - 0.9),
            xytext=(main_x, y_positions_main[5] + 0.9),
            arrowprops=dict(arrowstyle="->"))

# Flechas hacia cajas laterales
# Duplicates removed
ax.annotate("", xy=(main_x + box_width/2, side_boxes["Duplicates removed\n(n = 262)"]),
            xytext=(side_x - box_width/2, side_boxes["Duplicates removed\n(n = 262)"]),
            arrowprops=dict(arrowstyle="->"))

# Records excluded
ax.annotate("", xy=(main_x + box_width/2, side_boxes["Records excluded\n(n = 649)"]),
            xytext=(side_x - box_width/2, side_boxes["Records excluded\n(n = 649)"]),
            arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()
