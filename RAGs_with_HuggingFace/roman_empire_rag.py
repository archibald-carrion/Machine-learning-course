from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import numpy as np
import textwrap

# Source text about the Roman Empire
source_text = """
The Roman Empire was one of the largest and most influential empires in world history. At its height during the 2nd century AD, it covered over 5 million square kilometers and ruled over an estimated 70 million people, which was about 21% of the world's population at the time. The empire stretched from Britain in the northwest to Egypt in the southeast, encompassing the Mediterranean world and beyond.

Founded in 27 BCE when Octavian, the adopted son of Julius Caesar, was granted the title of Augustus by the Roman Senate, the Roman Empire succeeded the Roman Republic that had existed for nearly 500 years. Augustus established a system of government known as the Principate, which maintained the facade of the republic while concentrating power in the hands of the emperor.

The city of Rome was the capital of the empire and served as its political, economic, and cultural center. Rome was known for its impressive architectural achievements, including the Colosseum, the Pantheon, and the extensive system of aqueducts, roads, and public baths. The city's population might have exceeded one million inhabitants during its peak.

Roman society was highly stratified, with distinct social classes including patricians (aristocrats), plebeians (common citizens), freedmen (former slaves), and slaves. Latin was the official language of the empire, though Greek was widely spoken in the eastern provinces. Roman law, known for its sophistication and influence on modern legal systems, was codified in works such as the Twelve Tables and later the Corpus Juris Civilis under Emperor Justinian.

The Roman Empire's economy was based on agriculture, trade, and taxation. Roman currency, particularly the denarius, facilitated commerce throughout the empire. Romans built an extensive network of roads spanning approximately 250,000 miles, which facilitated trade, communication, and military movements across their vast territories.

Roman culture has had a profound influence on Western civilization, particularly in areas such as art, architecture, language, literature, law, and engineering. Many Romance languages, including Italian, French, Spanish, Portuguese, and Romanian, evolved from Latin. Roman architectural principles continued to be used long after the fall of the empire.

The western part of the Roman Empire collapsed in 476 CE when the last Roman emperor in the West, Romulus Augustulus, was deposed by the Germanic king Odoacer. The eastern half, known as the Byzantine Empire, continued to exist until 1453 CE when Constantinople fell to the Ottoman Turks.
"""

def split_text(text, chunk_size=300, overlap=50):
    """Splits the text into smaller chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def create_embeddings(chunks):
    """Generates embeddings for the text chunks using a Hugging Face model."""
    # Use a sentence transformer model for embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can experiment with other models
    embedding_model = pipeline("feature-extraction", model=model_name)
    embeddings = embedding_model(chunks)
    # The output is a list of lists; convert it to a NumPy array
    embeddings_array = np.array(embeddings).squeeze()
    return embeddings_array

def create_index(embeddings):
    """Creates a FAISS index for efficient similarity search."""
    dimension = embeddings.shape[1]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """Retrieves the most relevant text chunks for a given query."""
    # Use the same sentence transformer model for the query
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = pipeline("feature-extraction", model=model_name)
    query_embedding = embedding_model(query)
    query_embedding_array = np.array(query_embedding).squeeze().reshape(1, -1)  # Ensure correct shape for FAISS

    distances, indices = index.search(query_embedding_array, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_answer(question, context, model_name="distilbert-base-cased-distilled-squad"):
    """Generates an answer to the question using a Hugging Face model and retrieved context."""

    # Use a question answering pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
    )
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

def rag_pipeline(question, text):
    """
    Main function to orchestrate the RAG process.
    1. Splits the text into chunks
    2. Creates the embeddings for the chunks
    3. Creates a FAISS index
    4. Retrieves relevant chunks based on the query
    5. Generates the answer using a QA model
    """
    chunks = split_text(text)
    embeddings = create_embeddings(chunks)
    index = create_index(embeddings)
    relevant_chunks = retrieve_relevant_chunks(question, index, chunks)
    context = " ".join(relevant_chunks) # Join chunks into a single context string
    answer = generate_answer(question, context)
    return answer, context

def print_wrapped_text(text, width=80):
    """Prints the text wrapped to a specified width."""
    wrapped_text = textwrap.fill(text, width=width)
    print(wrapped_text)

if __name__ == "__main__":
    # Example usage
    question = "When was the Roman Empire founded?"
    answer, context = rag_pipeline(question, source_text)

    print("\nQuestion:")
    print_wrapped_text(question)
    print("\nAnswer:")
    print_wrapped_text(answer)
    print("\nContext Used:")
    print_wrapped_text(context)

    print("\n--- Another Example ---")
    question = "What was the capital city of the Roman Empire?"
    answer, context = rag_pipeline(question, source_text)
    print("\nQuestion:")
    print_wrapped_text(question)
    print("\nAnswer:")
    print_wrapped_text(answer)
    print("\nContext Used:")
    print_wrapped_text(context)