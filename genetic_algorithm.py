import numpy as np


class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight


def read_data(file_path):
    items = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        capacity = float(lines[0].split(":")[1].strip())
        for line in lines[1:]:
            if 'weight' in line:
                weight = float(line.split(":")[1].strip())
            elif 'value' in line:
                value = float(line.split(":")[1].strip())
                items.append(Item(value, weight))
    return capacity, items


class VectorChromosome:
    file_path = 'problem_plecakowy.txt'
    max_weight, items = read_data(file_path)

    def __init__(self, value=None):
        self.value = np.random.randint(2, size=(len(VectorChromosome.items))) if value is None else value

    def get_value(self):
        """
           Calculates and returns the total value of the selected items within this chromosome,
           subject to the constraint of not exceeding the maximum weight capacity.

           This method iterates over each gene in the chromosome, where each gene represents the inclusion
           (or exclusion) of an item in the knapsack. It computes the total weight and total value of the
           included items. If the total weight exceeds the maximum allowed weight (`self.max_weight`),
           the value of the chromosome is considered to be 0, indicating that the combination of items
           is not feasible under the given constraint. Otherwise, the total value of the included items
           is returned. This value serves as the fitness measure of the chromosome, with higher values
           being more desirable.

           Returns:
           - float: The total value of the selected items if the total weight is within the limit; otherwise, 0.

           Example:
           ```python
           chromosome = VectorChromosome()  # Assuming an appropriate constructor and initialization
           chromosome_value = chromosome.get_value()  # Calculate the fitness value of the chromosome
           print("Chromosome Value:", chromosome_value)
           ```
           """
        weight = sum(item.weight for item, selected in zip(VectorChromosome.items, self.value) if selected)
        value = sum(item.value for item, selected in zip(VectorChromosome.items, self.value) if selected)
        return value if weight <= self.max_weight else 0

    def sp_crossover(self, other):
        """
        Performs single-point crossover between two chromosomes.

        This method selects a random crossover point along the length of the chromosomes.
        It then creates a new chromosome (offspring) by combining the genes from one parent up to the crossover point
        and the genes from the other parent after that point. This simulates the crossover process in genetic recombination,
        which is a fundamental aspect of genetic algorithms, contributing to the generation of new offspring that inherit
        characteristics from both parents.

        Args:
        - other (VectorChromosome): The other parent chromosome to perform crossover with.

        Returns:
        - VectorChromosome: A new offspring chromosome resulting from the crossover.

        Example:
        ```python
        parent1 = VectorChromosome()
        parent2 = VectorChromosome()
        offspring = parent1.sp_crossover(parent2)  # The offspring will have parts of both parents
        ```
        """
        cross_point = np.random.randint(1, len(VectorChromosome.items))
        child_value = np.concatenate([self.value[:cross_point], other.value[cross_point:]])
        return VectorChromosome(value=child_value)

    def mutate(self, rate):
        """
        Performs mutation on the chromosome with a given probability.

        This method iterates through each gene in the chromosome and, with a probability defined by 'rate',
        inverts its value (from 0 to 1 or from 1 to 0). This simulates genetic mutations, introducing variability
        within the population, which is essential for the evolutionary process in genetic algorithms. Mutation helps
        to ensure genetic diversity and prevent premature convergence on suboptimal solutions.

        Args:
        - rate (float): The mutation rate, defining the probability of mutation for each gene, with a value between 0 and 1.

        Example:
        ```python
        chromosome = VectorChromosome()  # Assuming an appropriate constructor is available
        chromosome.mutate(0.05)  # There's a 5% chance for each gene to undergo mutation
        ```
        """
        if np.random.rand() < rate:
            to_mutate = np.random.randint(0, len(VectorChromosome.items))
            self.value[to_mutate] = 1 - self.value[to_mutate]

    def __str__(self):
        return str(self.value)


class VectorPopulation:

    def __init__(self, population_size, chromosomes=None):
        self.chromosomes: list[VectorChromosome] = [VectorChromosome() for _ in
                                                    range(population_size)] if chromosomes is None else chromosomes

    def get_best(self) -> VectorChromosome:
        """
           Identifies and returns the best chromosome in the population based on fitness values.

           This method first computes the fitness of each chromosome in the population by calling the `get_fits` method,
           which returns an array of fitness values. It then identifies the index of the highest fitness value in this array
           using NumPy's `argmax` function. The chromosome at this index is considered the best performing chromosome in terms
           of solving the problem or achieving the highest fitness. This best chromosome is often used for analysis, further
           evolutionary operations, or as a part of elitism strategy where it is directly passed to the next generation to ensure
           the retention of good solutions.

           Returns:
           - VectorChromosome: The chromosome with the highest fitness value in the current population.

           Example:
           ```python
           population_instance = VectorPopulation(population_size=100)  # Assuming an appropriate constructor is available
           best_chromosome = population_instance.get_best()  # Retrieve the best performing chromosome
           print("Best Chromosome Fitness:", best_chromosome.get_value())
           ```
           """
        fitness = self.get_fits()
        return self.chromosomes[fitness.argmax()]

    def get_fits(self):
        """
        Computes and returns the fitness values for all chromosomes in the population.

        This method iterates over each chromosome in the population, evaluates its fitness by calling
        the `get_value` method of each `VectorChromosome` instance, and then compiles these fitness values
        into a NumPy array. The fitness value of a chromosome is typically determined by how well it solves
        the problem at hand or how close it comes to the optimal solution. This array of fitness values is
        crucial for selecting chromosomes for crossover and mutation in the genetic algorithm process.

        Returns:
        - numpy.ndarray: An array of fitness values for each chromosome in the population.

        Example:
        ```python
        population_instance = VectorPopulation(population_size=100)  # Assuming an appropriate constructor is available
        fitness_values = population_instance.get_fits()  # Get the fitness values for the population
        ```
        """
        return np.array([ch.get_value() for ch in self.chromosomes])

    def __len__(self):
        return len(self.chromosomes)

    def get_elitism_next_generation(self, size):
        """
           Generates the next generation of chromosomes using the elitism approach.

           This method selects the 'size' number of best chromosomes (based on their fitness values)
           from the current population to form the core of the next generation. This elitism strategy ensures
           that the highest-quality individuals are preserved across generations, maintaining the best solutions
           found so far. However, this implementation removes one chromosome at a time from the original population
           and recalculates the best chromosome, which might not be the most efficient approach for large populations
           or high elitism sizes.

           Args:
           - size (int): The number of top-performing chromosomes to retain for the next generation.

           Returns:
           - VectorPopulation: A new population instance containing the elite chromosomes.

           Note:
           This method modifies the current population by repeatedly removing the least fit chromosome to find the next best.
           In large populations, this might be less efficient than sorting and selecting the top 'size' chromosomes directly.

           Example:
           ```python
           current_population = VectorPopulation(population_size=100)  # Assuming an appropriate constructor is available
           next_generation = current_population.get_elitism_next_generation(size=10)  # Retains the top 10 chromosomes
           ```
           """
        new_gen_chromosomes = []
        pop_ = self
        for _ in range(size):
            new_gen_chromosomes.append(pop_.get_best())
            new_chromosomes = self.chromosomes.copy()

            new_chromosomes.pop()
            pop_ = VectorPopulation(population_size=len(pop_.chromosomes) - 1, chromosomes=new_chromosomes)
        return pop_

    def add_vector(self, chromosome: VectorChromosome):
        """
        Adds a new chromosome to the population.

        This method appends a given chromosome (VectorChromosome instance) to the current list of chromosomes
        within the population. It's typically used during the generation of a new population, where offspring
        chromosomes created through crossover and mutation are added to form the next generation.

        Args:
        - chromosome (VectorChromosome): The chromosome to be added to the population.

        Example:
        ```python
        new_chromosome = VectorChromosome()  # Assuming an appropriate constructor is available
        population_instance = VectorPopulation(population_size=100)  # Create an instance of the population
        population_instance.add_vector(new_chromosome)  # Add the new chromosome to the population
        ```
        """
        self.chromosomes.append(chromosome)


class TournamentSelection:
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population, fits):
        """
        Selects a single chromosome from the population using tournament selection.

        This method randomly selects 'tournament_size' chromosomes from the population and then
        chooses the best chromosome among them based on their fitness values. Tournament selection
        is a stochastic method that provides a balance between exploration and exploitation by
        allowing less fit chromosomes a chance to be selected, while still favoring the fitter
        chromosomes. This method is particularly useful in maintaining diversity within the population
        and preventing premature convergence to suboptimal solutions.

        Args:
        - population (VectorPopulation): The current population from which to select a chromosome.
        - fits (array-like): An array of fitness values corresponding to each chromosome in the population.

        Returns:
        - VectorChromosome: The chromosome selected as the winner of the tournament.

        Example:
        ```python
        current_population = VectorPopulation(population_size=100)  # Assuming an appropriate constructor is available
        fits = current_population.get_fits()  # Obtain fitness values for the population
        tournament = TournamentSelection(tournament_size=5)
        selected_chromosome = tournament.select(current_population, fits)  # Selects a chromosome using tournament selection
        ```
        """
        selected_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        selected_fits = fits[selected_indices]
        winner_index = selected_indices[selected_fits.argmax()]
        return population.chromosomes[winner_index]


class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, elitism_size=16, max_iterations_without_improvement=10, mutation_rate=0.1):
        self.max_generations = max_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_size = int(elitism_size * population_size)
        self.population = VectorPopulation(population_size=population_size)
        self.selection = TournamentSelection()
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.iterations_without_improvement = 0
        self.best_fitness = 0

    def run(self):
        """
            Executes the genetic algorithm to evolve a population towards optimal solutions.

            This method iterates through a specified number of generations or until no improvement is
            observed for a predefined number of iterations. In each generation, it performs the following steps:
            1. Evaluate the fitness of each chromosome in the current population.
            2. Generate a new population by first including the best chromosomes (elitism),
               then filling the rest of the population with offspring created via crossover and mutation.
            3. Select parents for crossover using tournament selection to create offspring.
            4. Apply mutation to offspring with a specified mutation rate.
            5. Update the current population with the new population.
            6. If a new best solution is found, update the best solution and reset the counter for iterations without improvement.

            The algorithm terminates when the maximum number of generations is reached or if no improvement is found within
            the specified number of iterations without improvement.

            Returns:
            - best_solution (VectorChromosome): The best solution found throughout the run of the algorithm.
            - best_fitness (float): The fitness value of the best solution.

            Example:
            ```python
            ga = GeneticAlgorithm(population_size=100, max_generations=50, elitism_size=10, max_iterations_without_improvement=20)
            best_solution, best_fitness = ga.run()
            print(f"Best Fitness: {best_fitness}")
            ```
            """
        best_solution = None
        best_fitness = 0

        for generation in range(self.max_generations):
            fits = self.population.get_fits()
            new_population = self.population.get_elitism_next_generation(self.elitism_size)
            while len(new_population) < self.population_size:
                parent1 = self.selection.select(self.population, fits)
                parent2 = self.selection.select(self.population, fits)
                child = parent1.sp_crossover(parent2)
                child.mutate(self.mutation_rate)
                new_population.add_vector(child)

            current_best = new_population.get_best()
            current_best_value = current_best.get_value()

            if current_best_value > best_fitness:
                best_fitness = current_best_value
                best_solution = current_best
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1

            if self.iterations_without_improvement >= self.max_iterations_without_improvement:
                break

        return best_solution, best_fitness


file_path = 'problem_plecakowy.txt'
max_weight, items = read_data(file_path)
ga = GeneticAlgorithm(population_size=100, max_generations=20)
best_solution, best_fitness = ga.run()
print(f"Best Fitness: {best_fitness}")
