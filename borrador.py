#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: scrape_unemi_evaluacion_selenium.py
Descripción:
    Usa Selenium para autenticar en el SGA de UNEMI y extraer la tabla de evaluación docente.
Requisitos:
    - Chrome/Edge (Chromium) + ChromeDriver/EdgeDriver en PATH
    - pip install selenium
Uso:
    python scrape_unemi_evaluacion_selenium.py
"""

import sys
from getpass import getpass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def main():
    USER = "aavilesv"
    PASS = getpass("Contraseña UNEMI: ")

    # 1. Inicializar WebDriver (Chrome)
    driver = webdriver.Chrome()  # o webdriver.Edge()

    try:
        # 2. Intentar acceder a la página de evaluación (te redirigirá al login)
        target_url = (
            "https://sga.unemi.edu.ec/adm_evaluaciondocenteinvestigacioncoord"
            "?action=procesoevacoor"
            "&eva_id=OPPQQRRSSTTUUVVWWXXX"
            "&s=ALMEIDA+MONGE+ELKA+JENNIFER"
        )
        driver.get(target_url)

        # 3. Esperar a que aparezcan los campos de login y rellenarlos
        wait = WebDriverWait(driver, 10)

        # Ajusta estos selectores si los campos tienen otro name/id
        user_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
        pass_field = driver.find_element(By.NAME, "password")

        user_field.send_keys(USER)
        pass_field.send_keys(PASS)

        # Localiza y pulsa el botón de login
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type=submit]")
        login_button.click()

        # 4. Esperar a que volvamos al target_url (formulario ya no estará)
        wait.until(EC.url_contains("adm_evaluaciondocenteinvestigacioncoord"))

        # 5. Ahora extraer la tabla de datos
        # Ajusta el selector al ID/clase real de la tabla
        table = wait.until(EC.presence_of_element_located((By.ID, "tabla-datos")))

        # 6. Recorrer filas y celdas
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if not cols:
                continue
            values = [col.text.strip() for col in cols]
            print(values)

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
