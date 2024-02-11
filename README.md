# Genetic Algorithm for Knapsack Problem

This project contains a Python implementation of a Genetic Algorithm (GA) designed to solve the Knapsack Problem, a classic problem in combinatorial optimization. The algorithm evolves a population of potential solutions towards an optimal or near-optimal solution that maximizes the value of the items included in the knapsack without exceeding its weight capacity.

## Features

- **Item Representation**: Each item is characterized by its value and weight.
- **Chromosome Encoding**: Solutions are encoded as binary chromosomes, where each gene represents the presence or absence of an item in the knapsack.
- **Fitness Function**: The fitness of a chromosome is calculated based on the total value of the selected items, given the constraint of the maximum weight capacity.
- **Selection Mechanism**: Tournament selection is used to choose parent chromosomes for breeding.
- **Crossover Operator**: A single-point crossover mechanism is implemented to generate offspring from two parent chromosomes.
- **Mutation Operator**: Mutation is applied to offspring chromosomes to introduce variability, with a specified mutation rate.
- **Elitism**: The best-performing chromosomes are automatically passed to the next generation to ensure the retention of good solutions.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy library

### Installation

1. Clone the repository:
   ```
   git clone https://your-repository-url
   ```
2. Navigate to the project directory:
   ```
   cd genetic_algorithm_knapsack
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

To run the genetic algorithm with default settings:

```python
from genetic_algorithm import GeneticAlgorithm

# Define your items and maximum weight capacity
items = [Item(value=10, weight=5), Item(value=20, weight=10), ...]
max_weight = 50

# Initialize and run the genetic algorithm
ga = GeneticAlgorithm(items=items, max_weight=max_weight, population_size=100, max_generations=50)
best_solution, best_fitness = ga.run()

print(f"Best Fitness: {best_fitness}")
```

## Configuration

You can configure the genetic algorithm parameters including population size, number of generations, mutation rate, and tournament size for selection.

## Example

An example script `run_ga.py` is included to demonstrate how to set up and run the genetic algorithm on a sample set of items and a knapsack weight limit.

## Contributing

Contributions to improve the algorithm or add new features are welcome. Please follow the standard pull request process.

## Author

- **[Jakub Cymerman]** - *Initial work* - [jakub1090cn](https://github.com/jakub1090cn)

Feel free to contact me for any questions or feedback regarding this project.


## Acknowledgments

- Inspiration from classic genetic algorithm literature and the foundational work on combinatorial optimization problems.
