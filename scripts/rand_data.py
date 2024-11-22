import random
import sys

if len(sys.argv) != 2:
    print("Usage: python rand_data.py <n>")
    sys.exit(1)

n = sys.argv[1]
n = int(n)

while n > 0:
    x, y = random.random() * 10, random.random() * 10
    print(f"{x:.2}, {y:.2}")
    n -= 1
