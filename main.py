import time
import pandas
from transformers import BertTokenizer, BertModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy

input_file = 'input.csv'
output_file = '/app/bad_lines.csv'
prompts_file = '/app/prompts.txt'
embeddings_file = '/app/embeddings.npy'
bad_lines = []


def bad_line_handler(line):
    bad_lines.append(line)


df = pandas.read_csv(input_file, header=None, on_bad_lines=bad_line_handler, engine='python')
if bad_lines:
    with open(output_file, 'w') as f:
        for line in bad_lines:
            f.write(','.join(line) + '\n')
    print(f"Bad lines saved to {output_file}")

print("CSV file read successfully, bad lines handled.")

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
print("Model and tokenizer initialized.")


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


a_values = []
b_values = []
embeddings = []
prompts = []

for index, row in df.iterrows():
    a = row[0]
    b = row[1]
    prompt = f'Write one example of the company internal security rule for Software Development Department to correspond next instruction: <instruction>{a} {b}</instruction>'
    prompts.append(prompt)
    embedding = get_embeddings(prompt)
    a_values.append(a)
    b_values.append(b)
    embeddings.append(embedding)

numpy.save(embeddings_file, embeddings)
print(f"Embeddings saved to {embeddings_file}")

with open(prompts_file, 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')
print(f"Prompts saved to {prompts_file}")
print("Data prepared for insertion into Milvus.")


def connect_to_milvus(retries=5, delay=5):
    for i in range(retries):
        try:
            connections.connect(host="localhost", port="19530")
            print("Connected to Milvus")
            return True
        except Exception as e:
            print(f"Connection attempt {i + 1} failed: {e}")
            time.sleep(delay)
    return False


if not connect_to_milvus():
    print("Failed to connect to Milvus after several attempts.")
    exit(1)

fields = [
    FieldSchema(name="a", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="b", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Collection for storing prompts and embeddings")

collection_name = "prompts_collection"
if collection_name in connections.get_connection().list_collections():
    collection = Collection(name=collection_name)
else:
    collection = Collection(name=collection_name, schema=schema)

collection.insert([a_values, b_values, embeddings])

collection.load()

print("Data and embeddings have been successfully inserted into Milvus.")
