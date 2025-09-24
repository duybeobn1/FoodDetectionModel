from datasets import load_dataset

dataset = load_dataset("scuccorese/food-ingredients-dataset")
print(dataset['train'].features)