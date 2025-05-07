import matplotlib.pyplot as plt

# Crear figura
plt.figure(figsize=(10, 13))
steps = [
    ("Identification\nWoS + Scopus: 1,478 records", 0.9),
    ("After duplicates\n1,024 record", 0.7),
    ("Screening\nEmpirical heuristic: 1,024 records", 0.5),
    ("Excluded\nNon-experimental or ≤15 citations: 974", 0.3),
    ("Inclusion\nParadigmatic: 50 studies", 0.1)
]

# Dibujar cajas con flechas
for text, y in steps:
    plt.text(
        0.5, y, text,
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', pad=0.5),
        fontsize=10
    )
    # Flecha hacia el siguiente paso
    if y > 0.1:
        plt.annotate(
            "", xy=(0.5, y - 0.05), xytext=(0.5, y - 0.15),
            arrowprops=dict(arrowstyle="->", lw=1.2)
        )

plt.axis('off')
plt.tight_layout()
plt.show()
