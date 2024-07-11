import sys

r = open(sys.argv[1], 'r', buffering=16777216, encoding='utf-8', errors='replace')
w = open(sys.argv[2], 'w', buffering=16777216, encoding='utf-8', errors='replace')
for line in r:
    line = line.replace('�', '') 
    w.write(line)
r.close()
w.close();



#Generated with chatgpt: prompt: how to remove encoding error characters from a large file?
# and then it promptly tried to read the whole file at once and exploded
# so I referred to this (https://stackoverflow.com/questions/16669428/process-very-large-20gb-text-file-line-by-line) to try to fix things
# success tbd (it runs but i can't actually interact with the file meaningfully yet so) - 2024/07/08
