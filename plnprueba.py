# -*- coding: utf-8 -*-  # Define codificación del archivo para soportar caracteres especiales

"""
Dependencias (instalar):
pip install pandas numpy sentence-transformers torch openpyxl xlrd

OBJETIVO DEL SCRIPT
-------------------
Clasificar artículos (Scopus/WoS) por relevancia semántica respecto a un tema usando SBERT (Sentence Transformers).

ENTRADAS
--------
- Archivo CSV / XLS / XLSX con columnas:
  Title, Abstract, Author Keywords, Index Keywords

SALIDAS (en la misma carpeta del archivo de entrada)
----------------------------------------------------
- RESULTADO_FINAL_SBERT_PURO.xlsx con hojas:
  0_TODOS_CLASIFICADOS   -> TODOS los registros con score y decisión (aquí verás los ~140)
  1_LECTURA_PRIORITARIA  -> Alta relevancia + rescate desde Media
  2_AUDITORIA            -> Muestra aleatoria de excluidos para control PRISMA
  STATS                  -> Conteos por clase (ALTA/MEDIA/BAJA/DESCARTAR)

COLUMNAS NUEVAS QUE CREA
------------------------
- text_semantic   : Texto académico concatenado (Title + Abstract + Keywords)
- score_semantic  : Similaridad semántica (normalizada 0–1) del artículo vs TOPIC_TEXT
- FINAL_SCORE     : Score final (en SBERT puro es igual a score_semantic)
- DECISION        : Etiqueta por umbral: ALTA / MEDIA / BAJA / DESCARTAR
- Rescate_Reason  : Motivo de selección (Directo Alta / Rescate / No seleccionado)
- In_lectura      : True si el artículo quedó en la lista de lectura prioritaria
"""

# ===================== IMPORTS =====================

import pandas as pd  # Manejo de dataframes, lectura CSV/Excel, exportación Excel
import numpy as np  # Operaciones numéricas (normalización segura)
from pathlib import Path  # Manejo robusto de rutas
from sentence_transformers import SentenceTransformer, util  # Modelo SBERT y coseno
import time  # Medición de tiempo de ejecución

# ===================== CONFIG =====================

INPUT_FILE = r"G:\Mi unidad\2025\Master FRANCISCO MARCELO ALVARADO PORRAS\data\dataexcel.xlsx"  # Ruta del archivo de entrada

INPUT_PATH = Path(INPUT_FILE)  # Convertimos la ruta a Path para manipulación segura
if not INPUT_PATH.exists():  # Verificamos si el archivo existe
    raise FileNotFoundError(f"❌ El archivo no existe: {INPUT_PATH}")  # Si no existe, detenemos el script

# Guardar resultados EN LA MISMA CARPETA DEL ARCHIVO DE ENTRADA
OUTPUT_DIR = INPUT_PATH.parent  # Carpeta donde está el archivo de entrada
OUTPUT_EXCEL = OUTPUT_DIR / "RESULTADO_FINAL_SBERT_PURO34.xlsx"  # Excel final (misma carpeta)
OUTPUT_CSV_ALL = OUTPUT_DIR / "TODOS_CLASIFICADOS_SBERT.csv"  # CSV opcional con todo (misma carpeta)

TEXT_COLS = ["Title", "Abstract", "Author Keywords", "Index Keywords"]  # Columnas mínimas para construir texto académico
    
MODEL_NAME = "all-mpnet-base-v2"  # Modelo SBERT recomendado (muy fuerte para similitud semántica)

# Umbrales de clasificación (ajustados a temática CC.SS./Derecho)
TH_HIGH = 0.72  # >= 0.72 -> Alta relevancia
TH_MID  = 0.62  # >= 0.62 -> Media relevancia
TH_LOW  = 0.45  # >= 0.45 -> Baja relevancia; < 0.45 -> Descartar

# Texto “ancla” del tema: puede ser título + keywords o un mini-resumen del tema
TOPIC_TEXT = """
Climate Change and the Transformation of International Environmental Law:
Tensions, Justice, and Regulatory Challenges.
Keywords: international environmental law, climate change, climate justice,
global environmental governance, equity, international treaties,
human rights, sustainability, regulatory challenges.
"""

# Parámetros de selección
RESCUE_RATE = 0.15  # Porcentaje de rescate desde MEDIA (15%)
RESCUE_MIN = 5  # Rescate mínimo (si MEDIA es pequeña)
AUDIT_RATE = 0.10  # Porcentaje para auditoría (10% de excluidos)
AUDIT_MIN = 20  # Auditoría mínima
AUDIT_MAX = 50  # Auditoría máxima

# ===================== UTILIDADES =====================

def read_input_file(path: Path) -> pd.DataFrame:  # Función para leer CSV/XLS/XLSX
    """Lee automáticamente CSV / XLS / XLSX y devuelve DataFrame con dtype=str."""
    ext = path.suffix.lower()  # Extensión del archivo en minúsculas

    if ext == ".csv":  # Si es CSV
        # Nota: si tu CSV tuviera problemas de encoding, podrías añadir encoding="utf-8" o "latin1"
        return pd.read_csv(path, dtype=str)  # Leemos como texto para evitar errores de tipos

    if ext in [".xls", ".xlsx"]:  # Si es Excel
        return pd.read_excel(path, dtype=str)  # Leemos como texto

    raise ValueError("❌ Formato no soportado. Use .csv, .xls o .xlsx")  # Si no coincide, error claro

def safe_minmax(scores: np.ndarray) -> np.ndarray:  # Normalización min-max segura
    """Normaliza a [0,1]. Si max==min, devuelve ceros para evitar división por cero."""
    s_min = float(np.min(scores))  # Mínimo
    s_max = float(np.max(scores))  # Máximo
    denom = s_max - s_min  # Denominador
    if denom == 0.0:  # Si todos los scores son iguales
        return np.zeros_like(scores, dtype=float)  # Retornamos 0 para todos (no hay discriminación)
    return (scores - s_min) / denom  # Normalización estándar

# ===================== CARGA MODELO =====================

print("⏳ Cargando modelo SBERT...")  # Mensaje de progreso
model = SentenceTransformer(MODEL_NAME)  # Carga del modelo
print("✅ Modelo cargado.")  # Confirmación

# ===================== FUNCIONES SEMÁNTICAS =====================

def semantic_score(texts: pd.Series) -> np.ndarray:  # Calcula score SBERT
    """
    Calcula la similitud coseno entre cada texto del dataset y el TOPIC_TEXT.
    Devuelve scores normalizados [0,1].
    """
    ref_emb = model.encode(TOPIC_TEXT, convert_to_tensor=True)  # Embedding del tema (tensor)
    art_emb = model.encode(  # Embeddings de artículos
        texts.tolist(),  # Convertimos la serie en lista de strings
        convert_to_tensor=True,  # Devuelve tensores (más rápido)
        show_progress_bar=True  # Barra de progreso
    )
    raw_scores = util.cos_sim(art_emb, ref_emb).cpu().numpy().flatten()  # Coseno -> numpy -> vector
    return safe_minmax(raw_scores)  # Normalizamos a 0-1 de forma segura

def classify(score: float) -> str:  # Clasificación por umbrales
    """Asigna categoría de relevancia según umbrales definidos."""
    if score >= TH_HIGH:  # Alta
        return "🔥 ALTA RELEVANCIA"
    if score >= TH_MID:  # Media
        return "✅ MEDIA RELEVANCIA"
    if score >= TH_LOW:  # Baja
        return "⚠️ BAJA RELEVANCIA"
    return "❌ DESCARTAR"  # Descartar

# ===================== MAIN =====================

def main():  # Función principal
    start_time = time.time()  # Tomamos tiempo de inicio

    print("=" * 60)  # Línea decorativa
    print("📊 Bibliometric Semantic Screening (SBERT)")  # Título
    print("▶ Execution started...")  # Estado
    print("=" * 60)  # Línea decorativa

    print(f"📂 Archivo de entrada: {INPUT_PATH}")  # Mostramos el archivo de entrada
    print(f"📁 Carpeta de salida: {OUTPUT_DIR}")  # Mostramos carpeta de salida (misma que entrada)

    # ---------- Lectura ----------
    df = read_input_file(INPUT_PATH).fillna("")  # Leemos el archivo y reemplazamos NaN por ""

    # ---------- Validación de columnas ----------
    missing_cols = [c for c in TEXT_COLS if c not in df.columns]  # Detectamos columnas faltantes
    if missing_cols:  # Si falta alguna
        raise ValueError(f"❌ Faltan columnas requeridas: {missing_cols}")  # Detenemos con error claro

    # ---------- Construcción de texto académico ----------
    df["text_semantic"] = df[TEXT_COLS].agg(" ".join, axis=1)  # Concatenamos Title+Abstract+Keywords
    before_filter = len(df)  # Guardamos tamaño antes del filtro
    df = df[df["text_semantic"].str.len() > 50].copy()  # Filtramos registros demasiado cortos
    after_filter = len(df)  # Guardamos tamaño después del filtro

    print(f"🧹 Registros antes del filtro: {before_filter}")  # Reporte
    print(f"🧹 Registros después del filtro: {after_filter}")  # Reporte
    print(f"🗑️  Eliminados por texto corto: {before_filter - after_filter}")  # Reporte

    # ---------- Cálculo SBERT ----------
    print("🧠 Calculando similitud semántica (SBERT)...")  # Estado
    df["score_semantic"] = semantic_score(df["text_semantic"])  # Calculamos score semántico

    # ---------- Decisión ----------
    df["DECISION"] = df["score_semantic"].apply(classify)  # Etiqueta por umbral
    df["FINAL_SCORE"] = df["score_semantic"]  # Score final = score_semantic (SBERT puro)

    # ---------- Rescate_Reason (para TODOS) ----------
    df["Rescate_Reason"] = "No seleccionado"  # Valor por defecto para todos

    # ================= RESCATE DESDE MEDIA =================
    df_high = df[df["DECISION"] == "🔥 ALTA RELEVANCIA"].copy()  # Altas pasan directo
    df_mid = df[df["DECISION"] == "✅ MEDIA RELEVANCIA"].copy()  # Medias candidatas a rescate

    df_high["Rescate_Reason"] = "Directo: Alta Relevancia"  # Motivo para altas

    rescue_n = max(RESCUE_MIN, int(len(df_mid) * RESCUE_RATE))  # Cantidad a rescatar
    rescue = (  # Selección de rescate
        df_mid.sort_values("FINAL_SCORE", ascending=False)  # Ordenamos por score descendente
            .head(rescue_n)  # Tomamos top N
            .copy()  # Copiamos
    )
    rescue["Rescate_Reason"] = "Rescate: Media Relevancia Alta"  # Motivo para rescatados

    # Lista de lectura prioritaria = ALTA + RESCATE
    df_final = pd.concat([df_high, rescue]).drop_duplicates()  # Unimos y eliminamos duplicados

    # Marcamos quién quedó en lectura prioritaria
    df["In_lectura"] = df.index.isin(df_final.index)  # True si está en df_final

    # Actualizamos Rescate_Reason en el DF completo (para que quede registrado en TODOS CLASIFICADOS)
    df.loc[df_high.index, "Rescate_Reason"] = "Directo: Alta Relevancia"  # Marcamos altas
    df.loc[rescue.index, "Rescate_Reason"] = "Rescate: Media Relevancia Alta"  # Marcamos rescatados

    # ================= AUDITORÍA =================
    df_excluded = df[~df["In_lectura"]].copy()  # Excluidos = todo lo que no quedó en lectura
    audit_n = int(len(df_excluded) * AUDIT_RATE)  # 10% de excluidos
    audit_n = max(AUDIT_MIN, min(AUDIT_MAX, audit_n))  # Limitamos entre 20 y 50
    audit_n = min(audit_n, len(df_excluded))  # No podemos tomar más que los que existen
    audit_sample = df_excluded.sample(audit_n, random_state=42) if audit_n > 0 else df_excluded  # Muestra audit

    # ================= REPORTES EN CONSOLA =================
    print("\n📊 RESUMEN (conteos por clase):")  # Encabezado
    print(df["DECISION"].value_counts())  # Conteo por categoría

    print("\n📌 SELECCIÓN FINAL:")  # Encabezado
    print(f"   - Alta relevancia (directo): {len(df_high)}")  # Altas
    print(f"   - Media relevancia total:     {len(df_mid)}")  # Medias
    print(f"   - Rescatados desde Media:     {len(rescue)}")  # Rescatados
    print(f"   --------------------------------------------")  # Línea
    print(f"   TOTAL LECTURA PRIORITARIA:    {len(df_final)}")  # Total lectura
    print(f"   TOTAL AUDITORÍA:              {len(audit_sample)}")  # Total auditoría

    # ================= EXPORTACIÓN =================
    print("\n💾 Exportando resultados...")  # Estado

    # Ordenamos para que el Excel sea legible
    df_all_sorted = df.sort_values(["In_lectura", "FINAL_SCORE"], ascending=[False, False]).copy()  # Primero lectura, luego score
    df_final_sorted = df_final.sort_values(["FINAL_SCORE"], ascending=False).copy()  # Orden lectura por score

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:  # Abrimos writer Excel
        df_all_sorted.to_excel(writer, sheet_name="0_TODOS_CLASIFICADOS", index=False)  # TODOS (aquí verás 140, etc.)
        df_final_sorted.to_excel(writer, sheet_name="1_LECTURA_PRIORITARIA", index=False)  # Lectura prioritaria
        audit_sample.to_excel(writer, sheet_name="2_AUDITORIA", index=False)  # Auditoría
        df["DECISION"].value_counts().to_frame("Total").to_excel(writer, sheet_name="STATS")  # Stats

    # CSV adicional (opcional) con todos
    df_all_sorted.to_csv(OUTPUT_CSV_ALL, index=False, encoding="utf-8")  # Guardamos CSV con todo

    print(f"✅ Excel generado: {OUTPUT_EXCEL}")  # Confirmación
    print(f"✅ CSV generado:   {OUTPUT_CSV_ALL}")  # Confirmación

    # ================= TIEMPO TOTAL =================
    elapsed = time.time() - start_time  # Tiempo total
    minutes = int(elapsed // 60)  # Minutos
    seconds = elapsed % 60  # Segundos

    print("=" * 60)  # Línea
    print("✅ Pipeline completed successfully")  # Estado
    print(f"⏱ Total execution time: {minutes} min {seconds:.2f} sec")  # Tiempo final
    print("=" * 60)  # Línea

# ===================== EJECUCIÓN =====================

if __name__ == "__main__":  # Punto de entrada
    main()  # Ejecutamos main
# Imprimir explicación para el usuario
    print("\n" + "="*60)
    print("ℹ️  GLOSARIO: ¿Qué significa 'score_semantic'?")
    print("-" * 60)
    print("Es el puntaje de similitud generado por el modelo SBERT.")
    print("1. Se calcula la 'Similitud del Coseno' entre el artículo y tu Tema.")
    print("2. Se aplica una normalización Min-Max sobre este lote de datos:")
    print("   • 0.0 = El artículo MENOS parecido encontrado en este archivo.")
    print("   • 1.0 = El artículo MÁS parecido encontrado en este archivo.")
    print("="*60 + "\n")