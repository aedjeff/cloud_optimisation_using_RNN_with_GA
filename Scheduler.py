import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

class N2TCClassifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(20, activation='sigmoid', input_shape=(3,)),
            keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, tasks):
        if len(tasks) < 2:
            return
        data = np.array([[t.execution_time, t.cost, t.system_efficiency] for t in tasks])
        labels = np.array([self.assign_label(t) for t in tasks])  # Use meaningful labels

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

    def assign_label(self, task):
        # Example logic for assigning labels based on task attributes
        if task.weight < 2.5:
            return 0  # Class 0
        elif task.weight < 5:
            return 1  # Class 1
        else:
            return 2  # Class 2

    def classify_task(self, task):
        data = np.array([[task.execution_time, task.cost, task.system_efficiency]])
        return np.argmax(self.model.predict(data)) + 1  # Class 1, 2, or 3

def calculate_fitness(tasks, queue):
    fitness_value = sum(task.execution_time + task.cost + task.system_efficiency for task in tasks)
    if any(task in queue for task in tasks):
        return 0.9 * fitness_value
    return fitness_value

def initialize_population(tasks, population_size=500):
    return [random.sample(tasks, len(tasks)) for _ in range(population_size)]

def selection(population, fitness_function, queue):
    return sorted(population, key=lambda x: fitness_function(x, queue), reverse=False)[:len(population) // 2]

def crossover(parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1[:], parent2[:]  # Return copies if too small for crossover
    
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def mutate(chromosome, mutation_rate=0.05):
    if len(chromosome) < 2:  # Ensure there are at least two genes to swap
        return chromosome

    if random.random() < mutation_rate:
        gene_index, gene_index2 = random.sample(range(len(chromosome)), 2)
        chromosome[gene_index], chromosome[gene_index2] = chromosome[gene_index2], chromosome[gene_index]
    
    return chromosome

def schedule_tasks(tasks, queue, classifier):
    classifier.train(tasks)
    classified_tasks = {1: [], 2: [], 3: []}
    for task in tasks:
        classified_tasks[classifier.classify_task(task)].append(task)
    
    selected_tasks = classified_tasks[2] + classified_tasks[3]
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
    return list(set(best_solution))

def main():
    tasks = [Task(i, random.randint(1, 10), random.randint(1, 10), random.uniform(0.5, 1.5)) for i in range(10000)]
    waiting_queue = []
    classifier = N2TCClassifier()
    
    while tasks:
        best_schedule = schedule_tasks(tasks, waiting_queue, classifier)
        scheduled_tasks = set(best_schedule)
        
        tasks = [task for task in tasks if task not in scheduled_tasks]
        waiting_queue = tasks  # Remaining tasks go to the waiting queue
        print("Best Task Schedule:", best_schedule)

if __name__ == "__main__":
    main()
