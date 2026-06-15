# Measuring Model Accuracy: Distance Formulas

Learn about the role of distance functions in machine learning.

Machine learning is often used to predict numeric values. For example, economic models might try to forecast stock prices, and weather models might try to predict the temperature on a specific day.

Because these models can be incredibly influential, understanding how accurate they are is crucial. We need to know how "far off" the model's predictions are from the actual values. This is where the mathematical concept of distance comes into play, offering a quantitative measure of a model's accuracy.

## Different Kinds of Distance

Measuring distance may seem completely straightforward, but in practice there are actually many different kinds of distance.

As an example, take the problem of measuring the distance it takes to travel between two places in a city.

If we happen to have a helicopter, the shortest distance might just be the length of a straight line from point A to point B.

But if we are walking? Now the shortest distance has to take the structure of city blocks and buildings into account. Unless we have superpowers, we're going to have to take some turns that we wouldn't in a helicopter.

Distance, in other words, depends on context. The same is true for machine learning models: depending on the model, we might measure the "distance" between our prediction and what actually happens in different ways.

## Exploring Distance Measures in Machine Learning

In our next lesson, we'll dive deeper into various distance measures used in machine learning, including:

- **Euclidean Distance**: the most common distance formula, the length of a straight line between two points
- **Manhattan Distance**: the "city block" distance, useful in urban planning models
- **Hamming distance**: used to measure distance between words in natural language processing

As we explore these distance measures, we'll also gain practical experience with Python lists and mathematical libraries for Python like NumPy and SciPy. These skills are not only foundational for understanding machine learning but also essential for applying these concepts in real-world data science projects.

Happy coding!
