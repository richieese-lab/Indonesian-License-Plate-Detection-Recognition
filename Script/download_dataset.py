import os
import shutil
import zipfile

def download_dataset(target_dir="Dataset/Raw"):
    """
    Mengunduh dan mengekstrak Indonesian License Plate Dataset dari Kaggle API.
    Memerlukan akun Kaggle & credentials (~/.kaggle/kaggle.json atau ENV KAGGLE_USERNAME & KAGGLE_KEY).
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "Library 'kagglehub' belum terpasang. Silakan jalankan: pip install kagglehub"
        )

    print("Mengunduh dataset via Kaggle API...")
    dataset_name = "juanthomaswijaya/indonesian-license-plate-dataset"
    downloaded_path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset berhasil diunduh ke cache: {downloaded_path}")

    os.makedirs(target_dir, exist_ok=True)

    # Cek apakah ada file zip di folder download atau folder yang sudah terekstrak
    zip_files = [f for f in os.listdir(downloaded_path) if f.endswith(".zip")]
    if zip_files:
        for z in zip_files:
            zip_path = os.path.join(downloaded_path, z)
            print(f"Mengekstrak {z} ke {target_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
    else:
        # Jika kagglehub langsung mengunduh file hasil ekstraksi
        print(f"📋 Menyalin dataset dari cache ke {target_dir}...")
        for item in os.listdir(downloaded_path):
            src = os.path.join(downloaded_path, item)
            dst = os.path.join(target_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    print(f"Dataset siap di: {target_dir}")
    return target_dir

if __name__ == "__main__":
    download_dataset()
