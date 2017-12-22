import os

def findfiles(rootdir):
    for filename in os.listdir(rootdir):
        path = os.path.abspath(rootdir +'/'+filename)
        print('what is  ' + path +'?')
        if os.path.isfile(path):
            print('file: ' + path)
            yield path
        elif os.path.isdir(path):
            print('dir: '+path)
            yield from findfiles(path)





g = findfiles(r'C:\Users\ikamensh\game_nidolons')

