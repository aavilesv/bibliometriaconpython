import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURACIÓN DE ESTILO ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 2. DATOS ---
years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
mean_tc = np.array([61.73, 57.80, 60.19, 74.33, 60.44, 65.84, 69.81, 65.31, 67.25, 21.75, 16.56])
n_articles = np.array([62, 59, 72, 82, 78, 105, 133, 163, 181, 48, 54])

# --- 3. COLORES ---
color_bar = '#1f77b4' 
color_line = '#ff7f0e'

# --- 4. CREAR LA FIGURA (CAMBIO AQUÍ) ---
# Cambié dpi=300 a dpi=120 para que NO se vea gigante en tu pantalla.
fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=120) 

# --- 5. EJE IZQUIERDO: BARRAS ---
bars = ax1.bar(years, n_articles, color=color_bar, alpha=0.75, width=0.7, 
               label='Annual Production (N)', zorder=2)

ax1.set_ylabel('Number of Documents (N)', color=color_bar, fontsize=10, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_bar, labelsize=9)

# --- 6. EJE DERECHO: LÍNEA ---
ax2 = ax1.twinx() 
ax2.plot(years, mean_tc, color=color_line, marker='o', markersize=4, linewidth=2, 
         label='Average Citations (TC)', zorder=3)

ax2.set_ylabel('Average Citations per Document', color=color_line, fontsize=10, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_line, labelsize=9)

# --- 7. AJUSTES ESTÉTICOS ---
ax1.set_xlabel('Year', fontsize=10, fontweight='bold')
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45, fontsize=8)
ax1.set_xlim(2013.2, 2024.8)

ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray', zorder=0)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# --- 8. NOTA EXPLICATIVA ---
ax2.annotate('Citation time lag:\nRecent papers have had\nless time to accumulate citations.',
            xy=(2023.5, 20), xycoords='data',
            xytext=(2017.5, 35), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='gray', lw=0.8),
            fontsize=7, color='#444444', 
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.9))

# --- 9. LEYENDA ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=False, fontsize=8)

plt.tight_layout()

# --- 10. GUARDADO PROFESIONAL (NUEVO) ---
# Esta línea guarda la imagen en tu carpeta con calidad 300 DPI (para la revista)
# aunque en pantalla la veas pequeña.
plt.savefig("Figure1_Trends.png", dpi=300, bbox_inches='tight')

# Mostrar en pantalla (se verá normal ahora)
plt.show()