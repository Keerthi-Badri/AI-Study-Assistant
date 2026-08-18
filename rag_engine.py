import hashlib
import os

import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import streamlit as st
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "./chroma_db"
DISTANCE_THRESHOLD = 1.2

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

class RAGEngine:

    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

        if not self.pdf_file:
            raise ValueError("PDF path cannot be empty.")

        if not os.path.exists(self.pdf_file):
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_file}"
            )

        # Create a unique collection name for this PDF
        with open(self.pdf_file, "rb") as file:
            pdf_hash = hashlib.sha256(
                file.read()
            ).hexdigest()[:12]

        self.collection_name = (
            f"study_material_{pdf_hash}"
        )

        # Load embedding model
        self.embedding_model = load_embedding_model()
        # Open persistent Chroma
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )

    def index_pdf(self):
        """
        Process the PDF only if it has not
        already been indexed.
        """

        if self.collection.count() > 0:
            return {
                "status": "existing",
                "chunks": self.collection.count()
            }

        pdf = pymupdf.open(self.pdf_file)

        all_text = ""

        for page in pdf:
            all_text += page.get_text() + "\n"

        pdf.close()

        if not all_text.strip():
            raise ValueError(
                "No extractable text was found in the PDF."
            )

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(
            all_text
        )

        if not chunks:
            raise ValueError(
                "No text chunks were created."
            )

        # Create embeddings
        embeddings = self.embedding_model.encode(
            chunks
        )

        # Store in Chroma
        self.collection.upsert(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        return {
            "status": "indexed",
            "chunks": len(chunks),
            "pages": len(pdf)
        }

    def retrieve(
        self,
        query,
        n_results=3
    ):
        """
        Convert the query to an embedding and
        retrieve the most relevant chunks.
        """

        if not query.strip():
            return {
                "found": False,
                "chunks": [],
                "distances": []
            }

        query_embedding = (
            self.embedding_model
            .encode(query)
            .tolist()
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        if not documents or not distances:
            return {
                "found": False,
                "chunks": [],
                "distances": []
            }

        # Check best match
        best_distance = distances[0]

        if best_distance > DISTANCE_THRESHOLD:
            return {
                "found": False,
                "chunks": documents,
                "distances": distances
            }

        return {
            "found": True,
            "chunks": documents,
            "distances": distances
        }

    def get_context(
        self,
        query,
        n_results=3
    ):
        """
        Retrieve relevant chunks and combine them
        into one context string.
        """

        result = self.retrieve(
            query,
            n_results=n_results
        )

        if not result["found"]:
            return None

        return "\n\n".join(
            result["chunks"]
        )