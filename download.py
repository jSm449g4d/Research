import os
import urllib.request
import zipfile
import importlib

os.makedirs("datasets", exist_ok=True)
def dl_zip_x(url:str=""):
    urllib.request.urlretrieve(url,os.path.join("./datasets",os.path.basename(url)))
    with zipfile.ZipFile(os.path.join("./datasets",os.path.basename(url))) as z:
        z.extractall(os.path.join("./datasets"))
        
# DIV2K
dl_zip_x("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip")
dl_zip_x("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip")
importlib.import_module("div2kRotMake")

# Mars surface image (Curiosity rover) labeled data set
dl_zip_x("https://zenodo.org/record/1049137/files/msl-images.zip?download=1")
importlib.import_module("mlsChoose")
importlib.import_module("mlsRotMake")
