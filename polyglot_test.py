import polyglot
from polyglot.text import Text, Word

word = Word("Obama", language="en")
print("Neighbors (Synonms) of {}".format(word)+"\n"+"-"*30)
for w in word.neighbors:
    print("{:<16}".format(w))
print("\n\nThe first 10 dimensions out the {} dimensions\n".format(word.vector.shape[0]))
print(word.vector[:10])
