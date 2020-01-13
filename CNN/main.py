import string
import sys
f = open("./train.txt")

inc = 0
reviews = []
exclude = set(string.punctuation)
while (inc < 25000):
    s = f.readline()
    s = s.replace("<br />", ' ')
    s = (''.join(ch for ch in s if ch not in exclude)).lower()
    reviews.append(s)
    t = s.split()
    # sys.stdout.write("\r{0}>".format("="*inc))
    # sys.stdout.flush()
    inc += 1

print(t)
print(len(reviews[5]))
print(reviews[5])
f.close()
