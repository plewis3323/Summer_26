# Distance Formula — Representing Points

In this lesson, you will learn three different ways to define the distance between two points:

- Euclidean Distance
- Manhattan Distance
- Hamming Distance

Before diving into the distance formulas, it is first important to consider how to represent points in your code.

In this exercise, we will use a list, where each item in the list represents a dimension of the point. For example, the point (5, 8) could be represented in Python like this:

```python
pt1 = [5, 8]
```

Points aren't limited to just two dimensions. For example, a five-dimensional point could be represented as `[4, 8, 15, 16, 23]`.

Ultimately, we want to find the distance between two points. We'll be writing functions that look like this:

```python
distance([1, 2, 3], [5, 8, 9])
```

Note that we can only find the difference between two points if they have the same number of dimensions!
