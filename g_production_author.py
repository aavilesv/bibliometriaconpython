import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. CONFIGURACIÓN DE ESTILO ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 2. DATOS (Organizados en DataFrame para facilitar el ordenamiento) ---
data = {
    'Author': [
        'DEMURU PAOLO', 'GIL DE ZÚÑIGA HOMERO', 'PRIOR HÉLDER', 'AMADO ADRIANA', 
        'DE ARAÚJO BRUNO', 'FERRACIOLI PAULO', 'HALLIN DANIEL C.', 'HAUBER GABRIELLA', 
        'KLIMKIEWICZ BEATA', 'MARCOS-MARNE HUGO', 'MARQUES F. JAMIL', 'MIHELJ SABINA', 
        'RICCI PAOLO', 'ROTHBERG DANILO', 'VON BÜLOW MARISA', 'ŠTĚTKA VÁCLAV'
    ],
    'h_index': [3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'TC':      [22, 48, 96, 185, 70, 7, 7, 22, 7, 44, 36, 7, 15, 7, 30, 7],
    'NP':      [4, 4, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2],
    'PY_start': [2020, 2022, 2021, 2017, 2021, 2024, 2024, 2022, 2024, 2022, 2023, 2024, 2021, 2024, 2021, 2024]
}

df = pd.DataFrame(data)

# ORDENAMIENTO: Ordenamos por Total de Citas (TC) para que quede como un ranking visual
df = df.sort_values(by='TC', ascending=True)  # Ascending para que el mayor quede arriba en barh

# --- 3. COLORES ---
color_bar = '#1f77b4'  # Azul para Citas
color_dot = '#ff7f0e'  # Naranja para H-index

# --- 4. CREAR LA FIGURA ---
# Aumentamos la altura (figsize) porque son muchos autores
fig, ax1 = plt.subplots(figsize=(8, 8), dpi=120)

# --- 5. EJE PRINCIPAL (BARRAS HORIZONTALES - CITAS) ---
# Usamos TC para las barras porque es el dato que varía más (de 7 a 185)
bars = ax1.barh(df['Author'], df['TC'], color=color_bar, alpha=0.7, height=0.7, 
                label='Total Citations (TC)', zorder=2)

ax1.set_xlabel('Total Citations (TC)', color=color_bar, fontsize=10, fontweight='bold')
ax1.tick_params(axis='x', labelcolor=color_bar)

# --- 6. EJE SECUNDARIO (PUNTOS - H-INDEX) ---
# Creamos un eje gemelo que comparta el eje Y
ax2 = ax1.twiny() 

# Usamos un gráfico de dispersión (scatter) para el h-index
# Añadimos un pequeño desplazamiento visual si se desea, pero aquí lo dejamos alineado
ax2.scatter(df['h_index'], df['Author'], color=color_dot, s=100, edgecolors='white', linewidth=1.5,
            label='h-index', zorder=3)

# Ajustamos los límites del eje h-index para que los puntos no queden pegados
ax2.set_xlim(0, 5) 
ax2.set_xlabel('h-index Impact Factor', color=color_dot, fontsize=10, fontweight='bold')
ax2.tick_params(axis='x', labelcolor=color_dot)

# --- 7. ETIQUETAS DE DATOS (TEXTO) ---
# Añadimos el dato de NP (Número de Publicaciones) como texto al lado de la barra
for i, (tc_val, np_val) in enumerate(zip(df['TC'], df['NP'])):
    ax1.text(tc_val + 2, i, f'(NP: {np_val})', va='center', fontsize=8, color='#555555')

# --- 8. AJUSTES ESTÉTICOS ---
ax1.grid(axis='x', linestyle='--', alpha=0.3, color='gray', zorder=0)

# Quitar bordes innecesarios
ax1.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_color(color_dot) # Pintar borde superior naranja
ax1.spines['bottom'].set_color(color_bar) # Pintar borde inferior azul

# --- 9. LEYENDA ---
# Creamos una leyenda manual combinada
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color_bar, lw=4, label='Total Citations (TC)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dot, markersize=10, label='h-index')
]
ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)

plt.title('Top Authors by Citations & Impact (h-index)', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()

# --- 10. GUARDADO ---
plt.savefig("Figure2_Authors.png", dpi=300, bbox_inches='tight')
plt.show()