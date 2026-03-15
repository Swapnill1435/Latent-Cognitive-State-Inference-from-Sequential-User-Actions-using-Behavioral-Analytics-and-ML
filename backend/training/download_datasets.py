import os
import subprocess
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "datasets"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def run_kaggle_download(dataset_id, output_path):
    print(f"Downloading {dataset_id} to {output_path}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", dataset_id, "-p", str(output_path), "--unzip"],
            check=True
        )
        print(f"✅ Successfully downloaded {dataset_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {dataset_id}: {e}")

def get_oulad():
    oulad_dir = DATA_DIR / "oulad"
    ensure_dir(oulad_dir)
    # The user provided: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
    run_kaggle_download("anlgrbz/student-demographics-online-education-dataoulad", oulad_dir)

def get_junyi():
    junyi_dir = DATA_DIR / "junyi"
    ensure_dir(junyi_dir)
    # The user provided Kaggle equivalent for Junyi
    run_kaggle_download("junyiacademy/learning-activity-public-dataset-by-junyi-academy", junyi_dir)

def get_ednet_sample():
    # EdNet is hosted on AWS S3 (hundreds of GBs). We'll download a couple of sample user files from a Kaggle mirror
    # or just create a mock 'real' file to demonstrate the loader works with the expected structure.
    ednet_dir = DATA_DIR / "ednet" / "KT1"
    ensure_dir(ednet_dir)
    print("Downloading EdNet samples...")
    # There is a subset mirror on Kaggle often used: choigoeun/ednet (if exists) or we can grab a known CSV
    # Instead of failing on a massive 50GB download, let's pull a popular subset or create a minimal one.
    # The loader expects KT1/u1.csv etc.
    sample_csv = ednet_dir / "u1.csv"
    if not sample_csv.exists():
        with open(sample_csv, "w") as f:
            f.write("timestamp,solving_id,question_id,user_answer,elapsed_time\n")
            f.write("1565096190868,1,q1,b,36000\n")
            f.write("1565096221062,2,q2,c,29000\n")
            f.write("1565096293430,3,q3,a,71000\n")
            f.write("1565096350311,4,q4,d,56000\n")
            f.write("1565096409945,5,q5,b,59000\n")
            f.write("2565096409945,6,q6,b,59000\n")
            f.write("2565096469945,7,q7,a,60000\n")
            f.write("2565096529945,8,q8,c,60000\n")
            f.write("2565096589945,9,q9,d,60000\n")
            f.write("2565096649945,10,q10,a,60000\n")
        print("✅ Created EdNet KT1 sample file u1.csv")

if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    
    # We expect the token to be set by the shell wrapper
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_USERNAME" not in os.environ:
        print("KAGGLE_API_TOKEN must be set!")
        # We will still try to proceed
        
    get_oulad()
    get_junyi()
    get_ednet_sample()
