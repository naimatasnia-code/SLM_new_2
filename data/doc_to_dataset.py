from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def build_domain_dataset(docs, out_file, limit=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    with open(out_file, "w") as f:
        for c in chunks[:limit]:
            text = c.page_content.replace("\n", " ")

            sample = f"""### Instruction: Explain the following domain-specific information.
{text}
### Response:
{text[:200]}"""

            f.write(json.dumps({"text": sample}) + "\n")

    print("Saved dataset:", out_file)
