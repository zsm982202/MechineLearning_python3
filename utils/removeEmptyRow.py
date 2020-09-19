import os
path = 'F:\\matlab_workspace\\test\MA\\'
name = 'maTest5.m'
f_r = open(path + name, encoding='UTF-8')
f_w = open(path + 'temp.m', 'w', encoding='UTF-8')
try:
    while True:
        line = f_r.readline()
        if len(line) == 0:
            break
        if line.count('\n') == len(line):
            continue
        f_w.write(line)
finally:
    f_r.close()
    f_w.close()

f_w = open(path + name, 'w', encoding='UTF-8')
f_r = open(path + 'temp.m', encoding='UTF-8')
try:
    while True:
        line = f_r.readline()
        if len(line) == 0:
            break
        if line.count('\n') == len(line):
            continue
        f_w.write(line)
finally:
    f_r.close()
    f_w.close()
    os.remove(path + 'temp.m')
