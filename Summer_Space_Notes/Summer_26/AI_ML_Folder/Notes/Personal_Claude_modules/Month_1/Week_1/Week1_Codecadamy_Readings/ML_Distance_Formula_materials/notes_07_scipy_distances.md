# Distance Formula — SciPy Distances

Now that you've written these three distance formulas yourself, let's look at how to use them using Python's SciPy library:

- Euclidean Distance: `.euclidean()`
- Manhattan Distance: `.cityblock()`
- Hamming Distance: `.hamming()`

There are a few noteworthy details:

1. **The scipy implementation of Manhattan distance is called `cityblock()`.** Remember, computing Manhattan distance is like asking how many blocks away you are from a point.
2. **The scipy implementation of Hamming distance will always return a number between 0 and 1.** Rather than summing the number of differences in dimensions, this implementation sums those differences and then divides by the total number of dimensions. For example, in your implementation, the Hamming distance between `[1, 2, 3]` and `[7, 2, -10]` would be 2. In scipy's version, it would be 2/3.

## Instructions

1. Call `distance.euclidean()` using the points `[1, 2]` and `[4, 0]` as parameters. Print the result.
2. Call `distance.cityblock()` using the points `[1, 2]` and `[4, 0]` as parameters. Print the result.
3. Call `distance.hamming()` using `[5, 4, 9]` and `[1, 7, 9]` as parameters and print the results. Your answer shouldn't match your function's results — scipy divides by the number of dimensions.

## Solution

```python
from scipy.spatial import distance

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

print(euclidean_distance([1, 2], [4, 0]))
print(manhattan_distance([1, 2], [4, 0]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))


print(distance.euclidean([1, 2], [4, 0]))
print(distance.cityblock([1, 2], [4, 0]))
print(distance.hamming([5, 4, 9], [1, 7, 9]))
```
