
import os
from urllib.request import urlopen
from tempfile import mkstemp


def get_module_from_github(url):
    """
    Loads a .py module from github (raw)
    Returns a module object
    """
    with urlopen(url) as response:
        if response.code == 200:
            text = str(response.read(), encoding="utf-8")
    
    _, path = mkstemp(suffix=".py", text=True)
    
    with open(path, mode='wt', encoding='utf-8') as fh:
        fh.write(text)
    
    directory, file_name = os.path.split(path)
    working_dir = os.getcwd()
    os.chdir(directory)
    module = __import__(file_name[:-3])
    os.chdir(working_dir)
    os.remove(path)
    return module



if __name__ == '__main__':
    #DEMO
    url = r"https://raw.githubusercontent.com/leztien/toy_datasets/master/make_decision_tree_data.py"
    module = get_module_from_github(url)
    func = module.make_decision_tree_data
    X,y = func(5,3,2)
    print(func, X, y, sep="\n\n")


