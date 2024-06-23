#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
my function to load a module from github
"""


def load_from_github(url):
    """
    Version 1.0
    older version that doesn't use tempfile
    """
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


def get_module_from_github(url):
    """
    Version 2.0
    Loads a .py module from github (raw)
    Returns a module object
    """
    import os
    from urllib.request import urlopen
    from tempfile import mkstemp
    
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


def load_module_from_github(url):
    """
    Version 3.0
    Loads a module from github (url to the raw file)
    returns a Python module.
    Copy this function into your code
    """
    from urllib.request import urlopen
    from tempfile import NamedTemporaryFile
    from os.path import split, dirname
    from sys import path
    
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"
    
    with NamedTemporaryFile(mode='w+b', suffix='.py') as f:
        f.write(obj.read())
        f.seek(0)
    
        path.append(dirname(f.name))
        module = __import__(split(f.name)[-1][:-3])
    del obj
    return module



if __name__ == '__main__':
    path = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_classification.py"
    module = load_module_from_github(path)
    X,y = module.make_data_for_classification(m=500, n=3, k=3, blobs_density=0.5)
    print(len(X))
    
    url = r"https://raw.githubusercontent.com/leztien/computer_science/main/priority_queue.py"
    module = load_module_from_github(url)
    Heap = module.Heap
    heap = Heap()
    heap.add(20); heap.add(30); heap.add(10)
    print(len(heap), heap.pop(), heap.top())
    
    
    



