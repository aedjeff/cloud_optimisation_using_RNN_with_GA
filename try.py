import pandas as pd 
import numpy as np 

''''''
# sample data generation
timestamps = pd.date_range(start="2025-03-01",periods=1000,freq="5T")
cpu_usage = np.random.uniform(low=20,high=80,size=1000)

data = pd.DataFrame({"timestamp":timestamps,"cpu_usage":cpu_usage})
data.to_csv("cpu_usage_data.csv",index=False)

data = pd.read_csv("cpu_usage_data.csv")
data.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['cpu_usage_scaled'] = scaler.fit_transform(data['cpu_usage'].values.reshape(-1,1))
data.head()

def create_sequences(data,seq_len):
  x,y=[],[]
  for i in range(len(data)-seq_len):
    x.append(data[i:i+seq_len])
    y.append(data[i+seq_len])
  return np.array(x),np.array(y)

seq_len=24

x,y=create_sequences(data['cpu_usage_scaled'],seq_len)

# model part 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(50, input_shape=(seq_len, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=20, batch_size=32)

model.save('cpu_predictor.h5')

# prediction part
import random
import numpy as np
from deap import base, creator, tools, algorithms

# Define fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize cost
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize GA
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 4)  # CPU cores (1 to 4)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)  # Individuals have 2 genes
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define evaluation function
def evaluate(individual):
    total_cores = sum(individual)  # Sum of CPU cores
    predicted_usage = model.predict(np.array([x[-1]]))  # Use RNN prediction
    cost = total_cores * 10  # Cost per CPU core (e.g., $10 per core)
    if predicted_usage > 0.9:  # Penalize if usage exceeds 90%
        cost += 1000  # Large penalty
    return (cost,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)  # Use two-point crossover
toolbox.register("mutate", tools.mutUniformInt, low=1, up=4, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

# Get the best solution
best_individual = tools.selBest(population, k=1)[0]
print("Optimal CPU cores:", best_individual)

