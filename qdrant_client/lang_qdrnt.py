import nbformat
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import time, os
from typing import List, Optional

from qdrant_client import QdrantClient


class CustomEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key="hHnQ6vbPMRKj7eCs5IG6QmjFTyYjVccW",
            base_url="https://api.deepinfra.com/v1/openai"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self.client.embeddings.create(
                model="BAAI/bge-m3",
                input=text,
                encoding_format="float"
            ).data[0].embedding
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]


class NotebookProcessor:
    def __init__(self):
        self.embeddings = CustomEmbeddings()

        client = QdrantClient(
            url="https://6e283805-9178-46dd-86bf-285943e9ffab.eu-west-1-0.aws.cloud.qdrant.io:6333",
            api_key="vKFvTPaqjPmHJFUr-9IN-hIG-cq27iMeqNTsN27gtIF6xVO5qlRueg"
        )

        collection_name = "ai-lab-academy-courses-bge"
        try:
            collection_info = client.get_collection(collection_name)
            print(f"Collection {collection_name} already exists")
        except Exception:
            from qdrant_client.http import models
            print(f"Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                ),
            )

        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        self.chunk_size = 512

    def extract_text_content(self, notebook) -> str:
        content = []
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                text = cell.source
                if '![' not in text:
                    content.append(f"Markdown:\n{text}")
            elif cell.cell_type == 'code':
                content.append(f"Code:\n{cell.source}")
                if cell.outputs:
                    for output in cell.outputs:
                        if 'text' in output:
                            content.append(f"Output:\n{str(output.text)}")
        return "\n\n".join(content)

    def create_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_size += len(word.split())
            current_chunk.append(word)

            if current_size >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_notebook(self, notebook_path: str) -> bool:
        start_time = time.time()
        try:
            print(f"\nProcessing notebook: {notebook_path}")

            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            text_content = self.extract_text_content(nb)
            print(f"Total text length: {len(text_content)} characters")
            print(f"Total cells: {len(nb.cells)}")

            chunks = self.create_chunks(text_content)
            print(f"Created {len(chunks)} chunks")

            documents = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}")

                doc = Document(
                    page_content=chunk,
                    metadata={
                        "chunk_num": i,
                        "file": notebook_path
                    }
                )
                documents.append(doc)

            # Add documents in batch
            self.vector_store.add_documents(documents)

            duration = time.time() - start_time
            print(f"\nProcessing completed:")
            print(f"File: {notebook_path}")
            print(f"Time taken: {duration:.2f} seconds")
            print(f"Total chunks processed: {len(chunks)}")
            print("Status: Success")
            return True

        except Exception as e:
            print(f"Error processing notebook: {str(e)}")
            return False

    def process_notebooks_in_directory(self, directory_path: str):
        print(f"\nProcessing all notebooks in: {directory_path}")

        notebook_files = [f for f in os.listdir(directory_path) if f.endswith('.ipynb')]
        total_notebooks = len(notebook_files)

        print(f"Found {total_notebooks} notebooks to process")

        successful = 0
        failed = 0

        for idx, notebook_file in enumerate(notebook_files, 1):
            full_path = os.path.join(directory_path, notebook_file)
            print(f"\nProcessing notebook {idx}/{total_notebooks}: {notebook_file}")

            try:
                if self.process_notebook(full_path):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {notebook_file}: {str(e)}")
                failed += 1
                continue

            time.sleep(1)

        print("\n" + "=" * 50)
        print("Processing Summary:")
        print(f"Total notebooks found: {total_notebooks}")
        print(f"Successfully processed: {successful}")
        print(f"Failed to process: {failed}")
        print("=" * 50)

    def search_content(self, query_text: str, limit: Optional[int] = 3):
        try:
            print(f"\nSearching for: '{query_text}'")

            results = self.vector_store.similarity_search_with_score(
                query=query_text,
                k=limit
            )

            print("\nSearch Results:")
            print("-" * 50)

            for i, (doc, score) in enumerate(results, 1):
                print(f"\nResult {i}")
                print(f"Similarity Score: {score:.2f}")
                print(f"File: {doc.metadata['file']}")
                print(f"Chunk: {doc.metadata['chunk_num']}")
                print("Content:")
                print("-" * 20)
                print(doc.page_content)
                print("-" * 50)

        except Exception as e:
            print(f"Search error: {str(e)}")


def main():
    print("Starting notebook processing...")
    processor = NotebookProcessor()

    # print("\nStep 1: Processing notebooks")
    # notebooks_dir = "notebooks"
    # processor.process_notebooks_in_directory(notebooks_dir)

    print("\nStep 2: Testing search")
    processor.search_content("ნეირონული ქსელები")


if __name__ == "__main__":
    main()