# import sys

in_dir = '/Users/jiaguo/OneDrive - cumc.columbia.edu/AcademicYear/Spring 2020/W4995 Deep Learning/Project/Data/3000/symbols.txt'
out_dir = '/Users/jiaguo/OneDrive - cumc.columbia.edu/AcademicYear/Spring 2020/W4995 Deep Learning/Project/Data/stocks_symbols.vocab'

with open(in_dir, 'r', encoding='utf-8') as fin:
    with open(out_dir, 'w', encoding='utf-8') as fout:
        for line in fin:
            idx = line.rfind(' ')
            word = line[:idx]
            fout.write(word+'\n')