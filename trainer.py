'''

Author: Eric Denovitzer
~ python <file_name> <limit_examples> <ratio>


'''
from ReviewTrainer import *
import sys
import json

def load(fname, prep, limit=0):
	f = open(fname)
	results = {}
	ex_proc = 0
	for line in f:
		js = json.loads(line)
		for k,v in prep(js).items():
			try:
				results[k].append(v)
			except KeyError:
				results[k] = [v]
		ex_proc += 1
		if ex_proc % 100 == 0: print 'Examples processed: %d' % ex_proc
		if limit != 0 and ex_proc > limit: break
	
	return results

def main():
	total = int(sys.argv[2])
	ratio = float(sys.argv[3])

	train = int(round(total*ratio))
	rv_trainer = ReviewTrainer()
	data = load(sys.argv[1], rv_trainer.preprocess, total)
	x = data['text']
	y = data['votes']
	rv_trainer.prepare_data(x[:train], y[:train])
	rv_trainer.train()
	pred = rv_trainer.predict(x[train:])
	print rv_trainer.get_error(pred, y[train:])

if __name__ == '__main__':
	main()
