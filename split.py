# 85981833 lines => ~10747729 = ~10000000 lines per file
# Goal of this is to split the cleaned file into 8/9 smaller (hopefully ~500MB) files, and then process each one individually
# this will be a little bit hardcoded to suit the particular file at hand bc i'm lazy (may change later but i think this way is faster anyways)
# nvm I lied I will be hardcoding it a lot
# forgot about the root node
# usage: python split.py "file to be split.xml" "outfile-prefix", should result in files named  outfile-prefix-0.xml,  outfile-prefix-1.xml,...  outfile-prefix-n.xml, without splitting up the records

import sys

r = open(sys.argv[1], 'r', buffering=16777216, encoding='utf-8') #the file should be clear of encoding errors at this point so
filelen = 10000000
filenum = 0
lines=0
head='''<?xml version="1.0"?>
<Records
    xmlns="http://pubchem.ncbi.nlm.nih.gov/pug_view"
    xmlns:xs="http://www.w3.org/2001/XMLSchema-instance"
    xs:schemaLocation="http://pubchem.ncbi.nlm.nih.gov/pug_view https://pubchem.ncbi.nlm.nih.gov/pug_view/pug_view.xsd"
 >
'''
w = open(sys.argv[2]+'-'+str(filenum)+".xml", 'w', buffering=16777216, encoding='utf-8')
for line in r:
    w.write(line)
    lines += 1
    if lines >= filelen and line.find('</Record>') != -1:
        w.write('</Records>');
        w.close()
        filenum += 1
        lines = 0
        w = open(sys.argv[2]+'-'+str(filenum)+".xml", 'w', buffering=16777216, encoding='utf-8')
        w.write(head)

w.close()
r.close()

        
