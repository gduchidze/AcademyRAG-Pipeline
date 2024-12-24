import nbformat
from openai import OpenAI
from pinecone import Pinecone
import time
import uuid
import os

class NotebookProcessor:
    def __init__(self):
        self.openai_client = OpenAI(
            api_key="hHnQ6vbPMRKj7eCs5IG6QmjFTyYjVccW",
            base_url="https://api.deepinfra.com/v1/openai"
        )
        self.pinecone_client = Pinecone(
            api_key="pcsk_5o8tFS_MiBEvmq8wNjo4GYvMB7yydCV1Gwaj4PT3b6VF7QzZaWKSR2QYb2F4axChcrkSj8"
        )
        self.index = self.pinecone_client.Index("ai-lab-academy-courses")
        self.chunk_size = 512

    def extract_text_content(self, notebook):
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

    def create_chunks(self, text):
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

    def process_notebook(self, notebook_path):
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

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}")

                embedding = self.openai_client.embeddings.create(
                    model="BAAI/bge-m3",
                    input=chunk,
                    encoding_format="float"
                ).data[0].embedding

                self.index.upsert(
                    vectors=[{
                        "id": str(uuid.uuid4()),
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "chunk_num": i,
                            "file": notebook_path
                        }
                    }],
                    namespace="notebooks"
                )

                time.sleep(0.5)

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

    def process_notebooks_in_directory(self, directory_path):
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

    def search_content(self, query_text, limit=3):
        try:
            print(f"\nSearching for: '{query_text}'")

            query_embedding = self.openai_client.embeddings.create(
                model="BAAI/bge-m3",  # Using the same model as original
                input=query_text,
                encoding_format="float"
            ).data[0].embedding

            results = self.index.query(
                namespace="notebooks",
                vector=query_embedding,
                top_k=limit,
                include_values=False,
                include_metadata=True
            )

            print("\nSearch Results:")
            print("-" * 50)

            for i, match in enumerate(results['matches'], 1):
                print(f"\nResult {i}")
                print(f"Similarity Score: {match['score']:.2f}")
                print(f"File: {match['metadata']['file']}")
                print(f"Chunk: {match['metadata']['chunk_num']}")
                print("Content:")
                print("-" * 20)
                print(match['metadata']['text'])
                print("-" * 50)

        except Exception as e:
            print(f"Search error: {str(e)}")


def main():
    print("Starting notebook processing...")
    processor = NotebookProcessor()

    print("\nStep 1: Processing notebooks")
    notebooks_dir = "notebooks"
    processor.process_notebooks_in_directory(notebooks_dir)

    # print("\nStep 2: Testing search")
    # processor.search_content("ნეირონული ქსელები")


if __name__ == "__main__":
    main()