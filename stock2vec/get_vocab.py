# get vocabulary of all symbols of stocks

specials = ['<s>', '<pad>', '</s>', '<unk>']

in_dir = './Data/symbols.txt'
out_dir = './Data/stocks_symbols.vocab'

with open(in_dir, 'r', encoding='utf-8') as fin:
    with open(out_dir, 'w', encoding='utf-8') as fout:
        for word in specials:
            fout.write(word+'\n')
        for line in fin:
            idx = line.rfind(' ')
            word = line[:idx]
            fout.write(word+'\n')
