from flask import Flask, request, jsonify
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from sentence_transformers import SentenceTransformer
import ollama
import json, uuid

app = Flask(__name__)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = embedding_model.encode(input)
        return batch_embeddings.tolist()

embed_fn = MyEmbeddingFunction()

client = PersistentClient(path="./chromadb")

collection = client.get_or_create_collection(
    name=f"openml-translations"
)

with open('data/data.jsonl') as f:
    for line in f:
        pair=json.loads(line)
        inp=pair['input'] 
        output=pair['output']
        embedding = embedding_model.encode(inp)

        # upsert to chromadb
        collection.upsert(
            ids=[str(uuid.uuid1())],
            metadatas=[dict(translation=output)],
            documents=[inp],
            embeddings=embedding.tolist(),
        )

def question(q):
    return """
    [INST] 
    <<SYS>> 
    You are used by a software package as an API. So you should only respond with a direct answer.
    
    You should translate English sentences into sentences of a reduced English vocabulary, such that these result sentences can serve as input for an AutoML system.
The reduced vocabulary consists of the following tokens: load, dataset, calculate, volume, cluster, clustering, regression, data, count, clusters, a, and, perform, using, id, surface, height, mass, density, columns.

The sentences are queries or instructions regarding a dataset consisting of four columns: id (integer), surface (in square meters, float), height (in meters, float), mass (in kilograms, float). The 'id', 'surface', 'height' and 'mass' tokens of the reduced vocabulary correspond to these column names.
These columns describe the id, surface, height and mass of each object. The volume of the object is not a column but can be calculated as the product of the surface and height. Density of the object can be calculated by dividing the mass by the volume of the object.

You shouldn't do any calculations, only translate sentences to the reduced vocabulary while considering the relations between the colums of the dataset, as detailed out in the previous 2 sentences.
From the input sentence you should deduce the type of machine learning algorithm. It should be either regression, clustering or none at all. 

As an example, a sentence such as `Given the objects dataset, how many types of objects can be found, according to their volume?` would be translated into `load dataset, calculate volume using the surface and height columns, cluster and count clusters`. In this case the required ML algorithm is clustering as there is no column indicating the object type. Instead, the object type is deduced by performing clustering, a type of unsupervised learning.
Another example:  "Train a model that can estimate density based on surface and mass given the objects dataset." is translated into "load dataset, calculate density using surface and mass columns, and perform regression". In this case the required ML algorithm is regression as density is an object property that is calculated by other columns and can thus be learned by regression, a supervised learning algorithm. 
    <</SYS>>
    

Can you translate the sentence "%s" and reply only with the translation?
[/INST]
""" %(q)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    q = data.get('question')
    similar_docs= collection.query(
        query_texts=[q],
        n_results=6,
    )
    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': question(q)}])
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)