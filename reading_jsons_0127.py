import codecs
import json

filis=[
'/Users/ps22344/Downloads/coursera/0.json',
'/Users/ps22344/Downloads/coursera/1e2.json',
'/Users/ps22344/Downloads/coursera/1e3.json',
'/Users/ps22344/Downloads/coursera/1e5.json',
'/Users/ps22344/Downloads/coursera/4.json',
'/Users/ps22344/Downloads/coursera/10.json']


def finder(input_file, word_list):
	"""
	looks for words in word_list in the input_file.
	"""
	with codecs.open(input_file, "r", "utf-8") as jsonin:
		dicti=json.load(jsonin)
	dicti={k:v for k,v in dicti.items() if k in ['bad','good']}
	for key in dicti:
		print [w for w in word_list if w in [word for word,coef in dicti[key]]]

for fi in filis:
	finder(fi, ['love', 'disappointed', 'great', 'money', 'quality'])