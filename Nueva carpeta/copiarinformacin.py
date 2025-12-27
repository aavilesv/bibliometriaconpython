import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
#   PANEL DE CONTROL
# ==========================================

ANCHO = 16
ALTO = 26  # Altura segura

# POSICIONES
POS_IZQ = 5.0      # Columna Izquierda (Flujo principal)
POS_DER = 11.0     # Columna Derecha (Exclusiones)
ANCHO_CAJA = 4.5
BYPASS_X = 14.5    # Línea vertical derecha

# ESTILOS FINALES (VERDE)
COLOR_FINAL_FONDO = '#E8F5E9'
COLOR_FINAL_BORDE = '#2E7D32'
GROSOR_FINAL = 3.5

# ==========================================

fig, ax = plt.subplots(figsize=(ANCHO, ALTO)) 
ax.set_xlim(0, ANCHO)
ax.set_ylim(0, ALTO)
ax.axis('off')

# --- FUNCIONES ---

def draw_box(x, y_top, text, phase=None, final=False):
    # Altura dinámica
    lines = text.count('\n') + 1
    h = 0.8 + lines * 0.5
    
    y_bottom = y_top - h
    y_center = y_top - (h/2)
    
    # Estilos
    if final:
        fc = COLOR_FINAL_FONDO; ec = COLOR_FINAL_BORDE; lw = GROSOR_FINAL
    else:
        fc = 'white'; ec = 'black'; lw = 1.2
        
    rect = patches.Rectangle((x, y_bottom), ANCHO_CAJA, h, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=10)
    ax.add_patch(rect)
    
    ax.text(x + ANCHO_CAJA/2, y_center, text, ha='center', va='center', fontsize=9.5, wrap=True, zorder=11, linespacing=1.4)
    
    # Etiqueta lateral (Fases)
    if phase:
        ax.plot([x - 1.5, x - 1.5], [y_top, y_bottom], color='#999', lw=2)
        ax.text(x - 1.8, y_center, phase, ha='center', va='center', 
                fontsize=11, fontweight='bold', rotation=90, color='#333')
        
    return y_bottom, y_center

def draw_arrow(x, y_start, y_end):
    if y_start > y_end:
        ax.arrow(x, y_start, 0, y_end - y_start, head_width=0.15, head_length=0.15, fc='k', ec='k', length_includes_head=True, zorder=5)

def draw_elbow(x_start, y_start, x_end, y_end):
    mid_x = (x_start + x_end) / 2
    ax.plot([x_start, mid_x], [y_start, y_start], 'k-', lw=1.2, zorder=5)
    ax.plot([mid_x, mid_x], [y_start, y_end], 'k-', lw=1.2, zorder=5)
    ax.arrow(mid_x, y_end, x_end - mid_x, 0, head_width=0.15, head_length=0.15, fc='k', ec='k', length_includes_head=True, zorder=5)

# =================== DIBUJO ===================

# Cabecera
ax.add_patch(patches.Rectangle((POS_IZQ - 0.5, 24.5), 11, 0.8, fc='#FFD700', ec='none'))
ax.text(POS_IZQ + 5, 24.9, "Identification of studies via databases and registers", ha='center', va='center', fontweight='bold', fontsize=12)

CURRENT_Y = 24.0
GAP = 1.2

# --- 1. IDENTIFICATION ---
# Caja 1
yb1, yc1 = draw_box(POS_IZQ, CURRENT_Y, "Records identified from:\nDatabases (n = 1,595)\n(WoS: 1,362; Scopus: 233)", phase="IDENTIFICATION")

# Caja 2 (YA SIN EL DETALLE DE DOI/FUZZY)
yb2, yc2 = draw_box(POS_DER, CURRENT_Y, "Records removed before screening:\nDuplicate records removed (n = 115)")

draw_elbow(POS_IZQ + ANCHO_CAJA, yc1, POS_DER, yc2)
NEXT_Y = min(yb1, yb2) - GAP
draw_arrow(POS_IZQ + ANCHO_CAJA/2, yb1, NEXT_Y)
CURRENT_Y = NEXT_Y

# --- 2. SCREENING ---
yb3, yc3 = draw_box(POS_IZQ, CURRENT_Y, "Records screened\n(n = 1,444)", phase="SCREENING")
yb4, yc4 = draw_box(POS_DER, CURRENT_Y, "Records excluded:\nIncomplete Metadata\n(n = 431)")
draw_elbow(POS_IZQ + ANCHO_CAJA, yc3, POS_DER, yc4)
NEXT_Y = min(yb3, yb4) - GAP
draw_arrow(POS_IZQ + ANCHO_CAJA/2, yb3, NEXT_Y)
CURRENT_Y = NEXT_Y

# --- 3. ELIGIBILITY ---
yb5, yc5 = draw_box(POS_IZQ, CURRENT_Y, "Reports assessed for eligibility\n(Valid Corpus)\n(n = 1,013)", phase="ELIGIBILITY")

# Linea Bypass
ax.plot([POS_IZQ + ANCHO_CAJA, BYPASS_X], [yc5, yc5], 'k--', lw=1.5) 

NEXT_Y = yb5 - GAP
draw_arrow(POS_IZQ + ANCHO_CAJA/2, yb5, NEXT_Y)
CURRENT_Y = NEXT_Y

yb6, yc6 = draw_box(POS_IZQ, CURRENT_Y, "Reports sought for retrieval\n(High Relevance ≥ 0.80)\n(n = 39)")
yb7, yc7 = draw_box(POS_DER, CURRENT_Y, "Records excluded by AI:\n(n = 974)\nDiscarded (<0.50): 424\nLow Rel.: 378 | Med Rel.: 172")
draw_elbow(POS_IZQ + ANCHO_CAJA/2, yc6 + 0.5, POS_DER, yc7)

NEXT_Y = min(yb6, yb7) - GAP
draw_arrow(POS_IZQ + ANCHO_CAJA/2, yb6, NEXT_Y)
CURRENT_Y = NEXT_Y

yb8, yc8 = draw_box(POS_IZQ, CURRENT_Y, "Reports assessed for\nfull-text eligibility (n = 39)")
yb9, yc9 = draw_box(POS_DER, CURRENT_Y, "Reports excluded:\nFull text not available\nOut of scope (n = 11)")
draw_elbow(POS_IZQ + ANCHO_CAJA/2, yc8 + 0.4, POS_DER, yc9)

# --- 4. INCLUDED ---
FINAL_Y_TOP = yb8 - 2.0 
draw_arrow(POS_IZQ + ANCHO_CAJA/2, yb8, FINAL_Y_TOP)

# Caja Final 1
ybf1, ycf1 = draw_box(POS_IZQ, FINAL_Y_TOP, "Studies included in\nSYSTEMATIC REVIEW\n(Qualitative Synthesis)\n(n = 28)", phase="INCLUDED", final=True)

# Caja Final 2
ybf2, ycf2 = draw_box(POS_DER, FINAL_Y_TOP, "Studies included in\nBIBLIOMETRIC ANALYSIS\n(Quantitative Trends)\n(n = 1,013)", final=True)

# Cierre Bypass
ax.plot([BYPASS_X, BYPASS_X], [yc5, ycf2], 'k--', lw=1.5)
ax.arrow(BYPASS_X, ycf2, POS_DER + ANCHO_CAJA - BYPASS_X, 0, head_width=0.15, fc='k', ec='k', length_includes_head=True)
ax.text(BYPASS_X + 0.4, (yc5 + ycf2)/2, "Total Valid Corpus included in Bibliometrics", 
        rotation=270, ha='center', va='center', fontsize=10, style='italic', backgroundcolor='white')

plt.tight_layout()
plt.savefig("PRISMA_Final_Clean.png", dpi=300, bbox_inches='tight')
plt.show()