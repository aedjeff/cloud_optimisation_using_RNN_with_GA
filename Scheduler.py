import random

class Task:
    def __init__(self, task_id, execution_time, cost, system_efficiency):
        self.task_id = task_id
        self.execution_time = execution_time
        self.cost = cost
        self.system_efficiency = system_efficiency
        self.weight = self.calculate_weight()

    def calculate_weight(self):
        weight_parameters = {"ET": 0.4, "C": 0.3, "SE": 0.3}  # Example weights
        return (weight_parameters["ET"] * self.execution_time +
                weight_parameters["C"] * self.cost +
                weight_parameters["SE"] * self.system_efficiency)

    def __repr__(self):
        return f"Task({self.task_id}, Weight: {self.weight:.2f})"

class TaskClassifier:
    def __init__(self, epsilon=0.1):
        self.classes = {1: [], 2: [], 3: []}
        self.epsilon = epsilon

    def classify_task(self, task, queue):
        for class_id in [1, 2, 3]:
            if self.classes[class_id]:
                avg_weight = sum(t.weight for t in self.classes[class_id]) / len(self.classes[class_id])
                if abs(task.weight - avg_weight) < self.epsilon:
                    if task in queue and class_id > 1:
                        self.classes[class_id - 1].append(task)
                        return class_id - 1
                    else:
                        self.classes[class_id].append(task)
                        return class_id
        i = random.choice([1, 2, 3])
        self.classes[i].append(task)
        return i

def calculate_fitness(tasks, queue):
    fitness_value = sum(task.execution_time + task.cost for task in tasks)
    if any(task in queue for task in tasks):
        return 0.9 * fitness_value
    return fitness_value

def initialize_population(tasks, population_size=500):
    return [random.sample(tasks, len(tasks)) for _ in range(population_size)]

def selection(population, fitness_function, queue):
    return sorted(population, key=lambda x: fitness_function(x, queue), reverse=True)[:len(population) // 2]

def crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def mutate(chromosome, mutation_rate=0.05):
    if random.random() < mutation_rate:
        gene_index = random.randint(0, len(chromosome) - 1)
        chromosome[gene_index] = random.choice(chromosome)  # Randomly change the gene content
    return chromosome

def schedule_tasks(tasks, queue):
    classifier = TaskClassifier()
    for task in tasks:
        classifier.classify_task(task, queue)
    
    selected_tasks = classifier.classes[2] + classifier.classes[3]
    population = initialize_population(selected_tasks)
    for _ in range(100):
        selected = selection(population, calculate_fitness, queue)
        new_population = selected
        for i in range(0, len(selected), 2):
            if i+1 < len(selected):
                child1, child2 = crossover(selected[i], selected[i+1])
                new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
    
    best_solution = selection(population, calculate_fitness, queue)[0]
    return best_solution


def main():
    tasks = [Task(i, random.randint(1, 10), random.randint(1, 10), random.uniform(0.5, 1.5)) for i in range(10)]
    waiting_queue = []
    
    while tasks:
        best_schedule = schedule_tasks(tasks, waiting_queue)
        scheduled_tasks = set(best_schedule)
        
        tasks = [task for task in tasks if task not in scheduled_tasks]
        waiting_queue = tasks  # Remaining tasks go to the waiting queue
        print("Best Task Schedule:", best_schedule)

if __name__ == "__main__":
    main()
