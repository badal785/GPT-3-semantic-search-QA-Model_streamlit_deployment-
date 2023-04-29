import streamlit as st
import openai
import pinecone

# Set up OpenAI credentials
openai.api_key = "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"
model = "text-davinci-002"

# Set up Pinecone credentials
pinecone.init(api_key="66111e66-303d-4058-a399-15ae59643d11", environment="us-east1-gcp")
index_name = 'openai-ml-qa'
index = pinecone.Index(index_name)

# Define a function to query the model and return the answer
def ask_question(question, context):
    # Build prompt with context and question
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Query OpenAI model
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=2000)
    answer = response.choices[0].text.strip()
    return answer

# Define a function to search Pinecone and return relevant context
def search_context(question):
    # Embed the question using OpenAI
    embed_model = "text-embedding-ada-002"
    res = openai.Embedding.create(input=[question], engine=embed_model)
    xq = res['data'][0]['embedding']
    # Query Pinecone
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [x['metadata']['answer'] for x in res['matches']]
    # Concatenate the contexts
    limit = 3750
    prompt_start = "Answer the question based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {question}\nAnswer:"
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = prompt_start + "\n\n---\n\n".join(contexts[:i-1]) + prompt_end
            break
        elif i == len(contexts)-1:
            prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
    return prompt

# Define the Streamlit app
#def main():
 #   st.title("JForce Q&A Model")
  #  # Get user input
   # question = st.text_input("Enter your question:")
    3# Perform search if question is not empty
    #if question:
     #   # Search for context using Pinecone
      #  context = search_context(question)
      # 3  Ask the question using OpenAI
        #answer = ask_question(question, context)
        # Display the answer
       # st.success(answer)


def main():
    st.title("JForce Q&A Model")
    # Get user input
    question = st.text_input("Enter your question:")
    # Add search button
    if st.button("Search"):
        # Perform search if question is not empty
        if question:
            # Search for context using Pinecone
            context = search_context(question)
            # Ask the question using OpenAI
            answer = ask_question(question, context)
            # Display the answer
            st.success(answer)

if __name__ == "__main__":
    main()


