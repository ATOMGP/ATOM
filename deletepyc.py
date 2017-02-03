import os

filelist = [ f for f in os.listdir(".") if f.endswith(".pyc") ]
for f in filelist:
    os.remove(f)
