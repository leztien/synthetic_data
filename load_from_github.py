"""my function to load a module from github"""

def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module


#demo:
path = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_classification.py"
module = load_from_github(path)
X,y = module.make_data_for_classification(m=500, n=3, k=3, blobs_density=0.5)
