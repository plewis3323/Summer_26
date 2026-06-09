# Distance Formula — Hamming Distance

Hamming Distance is another slightly different variation on the distance formula. Instead of finding the difference of each dimension, Hamming distance only cares about whether the dimensions are exactly equal. When finding the Hamming distance between two points, add one for every dimension that has different values.

Hamming distance is used in spell checking algorithms. For example, the Hamming distance between the word "there" and the typo "thete" is one. Each letter is a dimension, and each dimension has the same value except for one.

## Instructions

1. Below `manhattan_distance()`, define your function in the same way as before. It should be named `hamming_distance()` and have two parameters named `pt1` and `pt2`. Create a variable named `distance`, have it start at 0, and return it.
2. After defining `distance`, create a for loop to loop through the dimensions of each point. If the values at each dimension are different, add 1 to `distance`.
3. Print the Hamming distance between `[1, 2]` and `[1, 100]`. Print the Hamming distance between `[5, 4, 9]` and `[1, 7, 9]`.

## Solution

```python
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

def hamming_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    if pt1[i] != pt2[i]:
      distance += 1
  return distance


print(hamming_distance([1, 2], [1, 100]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))
```
