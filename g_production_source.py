import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe # IMPORTANTE: Para el efecto de borde blanco

# --- 1. CONFIGURACIÓN DE ESTILO ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 2. DATOS ---
data = {
    'Source': [
        'INT. JOURNAL OF PRESS/POLITICS', 'LATIN AMERICAN POLITICS & SOC.', 
        'BRAZILIAN JOURNALISM RESEARCH', 'COMMUNICATION AND SOCIETY', 
        'GOVERNMENT AND OPPOSITION', 'INT. JOURNAL OF COMMUNICATION', 
        'JOURNALISM PRACTICE', 'LATIN AMERICAN POLICY', 
        'MEDIA E JORNALISMO', 'POLITICS AND GOVERNANCE', 
        'SOCIAL SCIENCES', 'THESIS ELEVEN', 
        'ADMINISTRATION AND SOCIETY'
    ],
    'h_index': [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    'TC':      [115, 27, 15, 42, 27, 9, 71, 8, 10, 29, 32, 15, 13],
    'NP':      [4, 3, 2, 4, 2, 3, 2, 2, 5, 2, 2, 2, 1]
}

df = pd.DataFrame(data)
df = df.sort_values(by='TC', ascending=True)

# --- 3. COLORES ---
color_bar = '#008080'  # Teal (Verde azulado)
color_dot = '#d62728'  # Rojo

# --- 4. CREAR LA FIGURA ---
fig, ax1 = plt.subplots(figsize=(8.5, 7), dpi=120)

# --- 5. EJE PRINCIPAL (BARRAS - FONDO) ---
# zorder=1: Al fondo
bars = ax1.barh(df['Source'], df['TC'], color=color_bar, alpha=0.6, height=0.6, 
                label='Total Citations (TC)', zorder=1)

ax1.set_xlabel('Total Citations (TC)', color=color_bar, fontsize=10, fontweight='bold')
ax1.tick_params(axis='x', labelcolor=color_bar)

# --- 6. EJE SECUNDARIO (PUNTOS - CAPA MEDIA) ---
ax2 = ax1.twiny() 

# zorder=2: En medio (sobre las barras, bajo el texto)
ax2.scatter(df['NP'], df['Source'], color=color_dot, s=120, alpha=0.9, 
            edgecolors='white', linewidth=1.5,
            label='Number of Papers (NP)', zorder=2)

ax2.set_xlim(0, 6)
ax2.set_xlabel('Number of Papers (NP)', color=color_dot, fontsize=10, fontweight='bold')
ax2.tick_params(axis='x', labelcolor=color_dot)
ax2.set_xticks([1, 2, 3, 4, 5, 6]) 

# --- 7. ANOTACIONES (TEXTO - CAPA SUPERIOR) ---
# zorder=3: Arriba del todo
for i, (tc_val, h_val) in enumerate(zip(df['TC'], df['h_index'])):
    # Texto un poco separado de la barra
    text_obj = ax1.text(tc_val + 2, i, f'(h-idx: {h_val})', 
                        va='center', fontsize=8, color='#333333', fontweight='bold', zorder=3)
    
    # Efecto "Halo" blanco para que se lea si el punto rojo pasa por detrás
    text_obj.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

# --- 8. AJUSTES ESTÉTICOS ---
ax1.grid(axis='x', linestyle='--', alpha=0.3, color='gray', zorder=0)

ax1.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_color(color_dot)
ax1.spines['bottom'].set_color(color_bar)

# --- 9. LEYENDA ---
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color_bar, lw=4, alpha=0.6, label='Impact: Total Citations'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dot, markersize=10, label='Volume: N. Articles')
]
ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)

plt.title('Top Sources: Impact vs. Production', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()

# --- 10. GUARDADO ---
plt.savefig("Figure3_Sources_Fixed.png", dpi=300, bbox_inches='tight')
plt.show()