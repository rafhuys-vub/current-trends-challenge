import logging
from flask import Flask, request, jsonify
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from sentence_transformers import SentenceTransformer
import ollama
import json, uuid

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

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
    <s>[INST] <<SYS>> 
    You are used by a software package as an API. So you should only respond with a direct answer.
    
    You should translate English sentences into sentences of a reduced English vocabulary, such that these result sentences can serve as input for an AutoML system.
    The reduced vocabulary only consists of the following words: load, dataset, calculate, volume, cluster, clustering, regression, data, count, clusters, a, and, perform, using, id, surface, height, mass, density, columns.

    The sentences are queries or instructions regarding a dataset consisting of four columns: id (integer), surface (in square meters, float), height (in meters, float), mass (in kilograms, float). The 'id', 'surface', 'height' and 'mass' words of the reduced vocabulary correspond to these column names.
    These columns describe the id, surface, height and mass of each object. The volume of the object is not a column but can be calculated as the product of the surface and height. Density of the object can be calculated by dividing the mass by the volume of the object.

    You shouldn't do any calculations, only translate sentences to the reduced vocabulary while considering the relations between the colums of the dataset, as detailed out in the previous 2 sentences.
    From the input sentence you should deduce the type of machine learning algorithm. It should be either regression, clustering or none at all. 
    
    You should reply _only_ with the translated sentence using the reduced vocabulary. Do not include any additional text or instructions. Next are a few examples of how to translate the input sentence.
    <</SYS>>

    Given the objects dataset, how many types of objects can be found, according to their volume? [/INST] load dataset, calculate volume using the surface and height columns, cluster and count clusters. </s><s>[INST]
    Train a model that can estimate density based on surface and mass given the objects dataset.[/INST] load dataset, calculate density using surface and mass columns, and perform regression. </s><s>[INST]
    %s[/INST]
    """ %(q)\

def flatten(xxs):
    return [x for xs in xxs for x in xs]
    
def print_embeddings(documents, metadatas):
    examples=[]
    for doc, meta in list(zip(documents, metadatas)):
        examples.append("{doc} [/INST] {translation} </s><s>[INST]".format(doc=doc, translation=meta['translation']))
    return '\n'.join(examples)
    
def question_examples(q, exs):
    return """
    <s>[INST] <<SYS>> 
    You are used by a software package as an API. So you should only respond with a direct answer.
    
    You should translate English sentences into sentences of a reduced English vocabulary, such that these result sentences can serve as input for an AutoML system.
    The reduced vocabulary only consists of the following words: load, dataset, calculate, volume, cluster, clustering, regression, data, count, clusters, a, and, perform, using, id, surface, height, mass, density, columns.

    The sentences are queries or instructions regarding a dataset consisting of four columns: id (integer), surface (in square meters, float), height (in meters, float), mass (in kilograms, float). The 'id', 'surface', 'height' and 'mass' words of the reduced vocabulary correspond to these column names.
    These columns describe the id, surface, height and mass of each object. The volume of the object is not a column but can be calculated as the product of the surface and height. Density of the object can be calculated by dividing the mass by the volume of the object.

    You shouldn't do any calculations, only translate sentences to the reduced vocabulary while considering the relations between the colums of the dataset, as detailed out in the previous 2 sentences.
    From the input sentence you should deduce the type of machine learning algorithm. It should be either regression, clustering or none at all. 
    
    You should reply _only_ with the translated sentence using the reduced vocabulary. Do not include any additional text or instructions. Next are a few examples of how to translate the input sentence.
    <</SYS>>
    %s
    %s[/INST]
    """ %(exs, q)\

app.logger.info(f"Preheat embeddings with a dummy query")

collection.query(
        query_texts=["What is the height of the objects in the dataset?"],
        n_results=1,
    )

app.logger.info(f"loading routes")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    q = data.get('question')
    app.logger.info(f"Received question: {q}")
    result = collection.query(
        query_texts=[q],
        n_results=6,
    )
    print(result)
    docs = flatten(result['documents'])
    metadatas = flatten(result['metadatas'])
    examples_string = print_embeddings(docs, metadatas)
    
    app.logger.info(f"Found similar documents: {examples_string}")
    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': question_examples(q, examples_string)}])
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

    logging.getLogger().setLevel(logging.DEBUG)
