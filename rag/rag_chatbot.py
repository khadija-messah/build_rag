import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

def sp(arr, s_chunk):
    for i in range(0, len(arr), s_chunk):
        yield arr[i:i+s_chunk]


def my_prop_rag():
    # file = open("data/resume.pdf", "r")
    file = PdfReader("data/resume.pdf")
    # arr = file.read()
    print("number pages is : ",len(file.pages))
    page = file.pages[0]
    print("page is : \n",page)
    print("extacrt is \n",page.extract_text)
    # chunk = list(sp(arr, 50))
    # batch = ollama.embed(model='embeddinggemma', input=chunk)
    # print("ecrir votre question")
    # while True:
    #     qs = input()
    #     if (qs == 'bye'):
    #         break
    #     imb_question = ollama.embed(model='embeddinggemma', input=qs)
    #     v1 = np.array(batch['embeddings'])
    #     v2 = np.array(imb_question['embeddings'][0])
    #     v2 = v2.reshape(1, -1)
    #     similarity = cosine_similarity(v1, v2)
    #     similarity_1d = similarity.reshape(-1)
    #     best_index = similarity_1d.argmax()
    #     best_chunk = chunk[best_index]
    #     promt = f"""
    #     Answer ONLY using the context below.
    #     If the answer is not in the context, say "I don't know".
    #     Context:
    #     {best_chunk}
    #     Question:
    #     {qs}
    #     """
    #     result = ollama.generate(model="mistral", prompt=promt)
    #     print(result["response"])
    file.close()


my_prop_rag()
