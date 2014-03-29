import sys
import json
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords

def load(fname, prep):
	f = open(fname)
	results = []
	for line in f:
		js = json.loads(line)
		res = prep(js)
		print res

	return results

def rev_process(l):
	return (l['text'], l['votes']['useful'])

def main():
	data = load(sys.argv[1], rev_process)

if __name__ == '__main__':
	main()
