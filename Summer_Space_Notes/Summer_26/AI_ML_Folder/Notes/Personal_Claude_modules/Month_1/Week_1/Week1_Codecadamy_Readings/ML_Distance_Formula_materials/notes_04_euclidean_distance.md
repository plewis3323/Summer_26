# Distance Formula — Euclidean Distance

Euclidean Distance is the most commonly used distance formula. To find the Euclidean distance between two points, we first calculate the squared distance between each dimension. If we add up all of these squared differences and take the square root, we've computed the Euclidean distance.

The equation:

$$\sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \ldots + (a_n - b_n)^2}$$

## Visual (from lesson image)

The lesson image shows two points plotted on a grid: $(b_1, b_2)$ at the lower left and $(a_1, a_2)$ at the upper right. A solid diagonal line labeled $d$ connects them (the Euclidean distance). Dashed lines form a right triangle: the horizontal leg is labeled $(a_1 - b_1)$ and the vertical leg is labeled $(a_2 - b_2)$.

$$d = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2}$$

(This is the Pythagorean theorem — the straight-line distance is the hypotenuse.)

## Instructions

1. Create a function named `euclidean_distance()` that takes two lists as parameters named `pt1` and `pt2`. In the function, create a variable named `distance`, set it equal to 0, and return `distance`.
2. After defining `distance`, create a for loop to loop through the dimensions of each point. Add the squared difference between each dimension to `distance`. (In Python, square `num` with `num ** 2`.)
3. Outside of the for loop, take the square root of `distance` and return that value.
4. Print the Euclidean distance between `[1, 2]` and `[4, 0]`. Add another print statement showing the Euclidean distance between `[5, 4, 3]` and `[1, 7, 9]`. Why can't you find the difference between `[2, 3, 4]` and `[1, 2]`?

## Solution

```python
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i])**2
  return distance**0.5

print(euclidean_distance([1, 2], [4, 0]))
print(euclidean_distance([5, 4, 3], [1, 7, 9]))
```

Note: you can't find the distance between `[2, 3, 4]` and `[1, 2]` because the points have different numbers of dimensions.
