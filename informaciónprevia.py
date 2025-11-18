import matplotlib.pyplot as plt
import numpy as np

# --- 1. Datos ACTUALIZADOS (Revistas/Journals) ---
# (en el orden que aparecen en la imagen, de arriba a abajo)

journals_top_to_bottom = [
    'Automatica',
    'IEEE Transactions on automatic control',
    'International journal of control',
    'IEEE-ASME Transactions on mechatronics',
    'Systems & control letters',
    'International journal of dynamics and control',
    'IEEE Transactions on control systems technology',
    'IEEE Transactions on industrial electronics',
    'IET Control theory and applications',
    'International journal of robust and nonlinear control'
]

# Valores para la parte azul ("Before 2023")
before_2023_top_to_bottom = [20, 13, 6, 6, 6, 3, 3, 4, 3, 3]

# Valores para la parte naranja ("Between 2023 - 2024")
between_2023_2024_top_to_bottom = [5, 4, 4, 0, 0, 2, 1, 0, 1, 0]

# Porcentajes mostrados en el gráfico
percentages_top_to_bottom = ['20%', '24%', '40%', '0%', '0%', '40%', '25%', '0%', '25%', '0%']

# --- 2. Preparar datos para Matplotlib ---
# Invertir listas para que matplotlib dibuje de arriba a abajo
journals = list(reversed(journals_top_to_bottom))
before_2023 = list(reversed(before_2023_top_to_bottom))
between_2023_2024 = list(reversed(between_2023_2024_top_to_bottom))
percentages = list(reversed(percentages_top_to_bottom))

# Convertir a arrays de NumPy para sumar
before_2023_np = np.array(before_2023)
between_2023_2024_np = np.array(between_2023_2024)
totals = before_2023_np + between_2023_2024_np

# --- 3. Crear el gráfico ---

# Definir los colores (los mismos que usaste)
color_blue = "#1f77b4"
color_orange = "#ff7f0e"

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 7)) # Ajusté un poco el alto para las 10 categorías

# Dibujar la primera barra (azul)
ax.barh(journals, before_2023, 
        label='Before 2023', 
        color=color_blue, 
        height=0.5,
        edgecolor='black')

# Dibujar la segunda barra (naranja), apilada
ax.barh(journals, between_2023_2024, 
        left=before_2023, 
        label='Between 2023 - 2024', 
        color=color_orange, 
        height=0.5,
        edgecolor='black')

# --- 4. Añadir anotaciones y etiquetas ---

# Añadir los porcentajes a la derecha de cada barra
for i in range(len(journals)):
    # Ajuste: Añado un pequeño espacio (0.5) desde el total para la etiqueta
    ax.text(totals[i] + 0.5, i, percentages[i], va='center', ha='left')

# --- 5. Personalizar el gráfico ---

# Título actualizado para coincidir con la imagen original
ax.set_title('sourceTitle bar trends graph', fontsize=14)

# Etiqueta del eje X (la que tenías es perfecta)
ax.set_xlabel('Total number of documents, with percentage of documents\npublished in the last years 2023 - 2024', fontsize=10)

# Cuadrícula vertical
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Ajustar el límite del eje X 
# El máximo total es 25 (Automatica), lo ajustamos
ax.set_xlim(0, max(totals) + 7) 

# Leyenda
ax.legend(loc='lower right')

# Ajustar diseño
plt.tight_layout()

# --- 6. Mostrar el gráfico ---
plt.show()