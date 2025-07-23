import pickle

with open("./20250722_214641_158005.pkl", "rb") as f:
    data = pickle.load(f)

print(data)

print("==============================================================================")

for key, value in data.items():
    print(key, value)