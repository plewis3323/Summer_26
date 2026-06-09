# Distance Formula — Manhattan Distance

Manhattan Distance is extremely similar to Euclidean distance. Rather than summing the squared difference between each dimension, we instead sum the absolute value of the difference between each dimension. It's called Manhattan distance because it's similar to how you might navigate when walking city blocks. If you've ever wondered "how many blocks will it take me to get from point A to point B", you've computed the Manhattan distance.

The equation:

$$|a_1 - b_1| + |a_2 - b_2| + \ldots + |a_n - b_n|$$

**Note: Manhattan distance will always be greater than or equal to Euclidean distance.**

## Visual (from lesson image)

The lesson image shows the same two points, $(b_1, b_2)$ at the lower left and $(a_1, a_2)$ at the upper right. Instead of a diagonal line, the path travels along the grid: a horizontal segment labeled $|a_1 - b_1|$ then a vertical segment labeled $|a_2 - b_2|$ — like walking city blocks.

$$d = |a_1 - b_1| + |a_2 - b_2|$$

## Instructions

1. Below `euclidean_distance()`, create a function called `manhattan_distance()` that takes two lists named `pt1` and `pt2` as parameters. In the function, create a variable named `distance`, set it equal to 0, and return it.
2. After defining `distance`, create a for loop to loop through the dimensions of each point. Add the absolute value of the difference between each dimension to `distance`. (In Python, take the absolute value of `num` with `abs(num)`.)
3. Below the print statements for Euclidean distance, print the Manhattan distance between `[1, 2]` and `[4, 0]`. Also print the Manhattan distance between `[5, 4, 3]` and `[1, 7, 9]`.

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
    distance += abs((pt1[i] - pt2[i]))
  return distance

print(euclidean_distance([1, 2], [4, 0]))
print(euclidean_distance([5, 4, 3], [1, 7, 9]))
print(manhattan_distance([1, 2], [4, 0]))
print(manhattan_distance([5, 4, 3], [1, 7, 9]))
```
