import openai
import pinecone

# Set up OpenAI credentials
openai.api_key = "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"
model = "text-embedding-ada-002"

# Set up Pinecone credentials
pinecone.init(api_key="66111e66-303d-4058-a399-15ae59643d11", environment="us-east1-gcp")

index_name = 'openai-ml-qa'

# Check if the index already exists (only create it if it doesn't)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        metric='cosine',
        metadata_config={'indexed': ['docs']}
    )

# Connect to the index
index = pinecone.Index(index_name)

# Define a function to insert data into the index
def insert_data(docs):
    # Create embeddings
    embeddings = []
    for doc in docs:
        res = openai.Embedding.create(input=[doc], engine=model)
        embeddings.append(res['data'][0]['embedding'])

    # Prepare data for Pinecone upsert
    data = []
    for i, embedding in enumerate(embeddings):
        data.append((str(i), embedding, {"doc": docs[i]}))

    # Upsert data to Pinecone
    index.upsert(vectors=data)

# Define a function to search the index
def search_index(query, top_k=3):
    # Create embedding for query
    res = openai.Embedding.create(input=[query], engine=model)
    query_embedding = res['data'][0]['embedding']

    # Search Pinecone index
    results = index.query(queries=[query_embedding], top_k=top_k)

    # Extract metadata and embeddings from results
    docs = []
    embeddings = []
    for result in results['matches']:
        docs.append(result.metadata['doc'])
        embeddings.append(result.embedding)

    return docs, embeddings
