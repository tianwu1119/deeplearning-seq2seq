# import sys

in_dir = '/Users/jiaguo/OneDrive - cumc.columbia.edu/AcademicYear/Spring 2020/W4995 Deep Learning/Project/Data/3000/new/symbols.txt'
out_dir = '/Users/jiaguo/OneDrive - cumc.columbia.edu/AcademicYear/Spring 2020/W4995 Deep Learning/Project/Data/3000/new/stocks_symbols.vocab'

specials = ['<SOS>', '<EOS>']

with open(in_dir, 'r', encoding='utf-8') as fin:
    with open(out_dir, 'w', encoding='utf-8') as fout:
        for word in specials:
            fout.write(word+'\n')
        for line in fin:
            idx = line.rfind(' ')
            word = line[:idx]
            fout.write(word+'\n')