# tile_preprocessor.py

import os
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# --- KONFIGURÁCIÓ ---
# Itt add meg a beállításokat, amikkel dolgozni szeretnél.
# -----------------------------------------------------------------------------

# 1. A mappa, ahol a nagy felbontású képeid vannak.
INPUT_FOLDER = "input_images"

# 2. A mappa, ahova a szkript menti a feldarabolt képeket (csempéket).
#    Ezt a mappát a szkript automatikusan létrehozza, ha nem létezik.
OUTPUT_FOLDER = "output_tiles"

# 3. A csempék mérete pixelben. Egy 512x512-es méret jó kiindulási alap.
#    A legtöbb modern GPU ezt a méretet hatékonyan kezeli.
TILE_SIZE = 512

# 4. Az átfedés mérete pixelben. Ez kritikus, hogy ne vágjunk ketté sejteket a csempék határán.
#    A csempeméret 10-20%-a általában jó érték (pl. 512-es méretnél 50-100 pixel).
OVERLAP = 100

# -----------------------------------------------------------------------------

def tile_image(image_path, output_dir, tile_size, overlap):
    """
    Ez a függvény egyetlen képet darabol fel átfedő csempékre.
    """
    # Kép beolvasása szürkeárnyalatosan
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [Hiba] A kép nem olvasható: {image_path}")
        return

    img_h, img_w = img.shape
    step = tile_size - overlap
    
    # Almappa létrehozása az adott kép csempéinek
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"  -> Darabolás {tile_size}x{tile_size} méretű csempékre, {overlap}px átfedéssel...")
    
    tile_count = 0
    # Végigiterálunk a képen a megadott lépésközzel
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            # A csempe kivágása
            tile = img[y:y + tile_size, x:x + tile_size]
            
            # Ellenőrizzük, hogy a csempe nem túl kicsi-e (a kép szélein fordulhat elő)
            # Ha a csempe magassága vagy szélessége kisebb, mint az átfedés, kihagyjuk.
            h, w = tile.shape
            if h < overlap or w < overlap:
                continue

            # A csempe elmentése egy beszédes névvel
            tile_filename = f"tile_y{y}_x{x}.png"
            output_path = os.path.join(output_dir, tile_filename)
            cv2.imwrite(output_path, tile)
            tile_count += 1
    
    print(f"  Sikeresen elmentve {tile_count} csempe a '{output_dir}' mappába.")


def main():
    """
    A fő program, ami végigmegy a bemeneti mappán és minden képre meghívja a daraboló függvényt.
    """
    print("Automatizált képdaraboló szkript elindult.")
    print(f"Bemeneti mappa: '{INPUT_FOLDER}'")
    
    # Ellenőrizzük, hogy a bemeneti mappa létezik-e
    if not os.path.isdir(INPUT_FOLDER):
        print(f"[HIBA] A bemeneti mappa ('{INPUT_FOLDER}') nem található! Hozd létre, és másold bele a képeidet.")
        return

    # Kimeneti főmappa létrehozása
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Létrehozva a kimeneti mappa: '{OUTPUT_FOLDER}'")

    # A bemeneti mappa fájljainak listázása
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print("Nem található képfájl a bemeneti mappában.")
        return
        
    print(f"Talált {len(image_files)} képfájl. A feldolgozás megkezdődik...")

    for filename in image_files:
        image_path = os.path.join(INPUT_FOLDER, filename)
        
        # Az almappa neve az eredeti fájlnév kiterjesztés nélkül
        base_filename = os.path.splitext(filename)[0]
        image_output_dir = os.path.join(OUTPUT_FOLDER, base_filename)
        
        print(f"\n[Feldolgozás alatt] '{filename}'")
        tile_image(image_path, image_output_dir, TILE_SIZE, OVERLAP)
        
    print("\nA feldarabolás befejeződött.")


if __name__ == "__main__":
    main()
