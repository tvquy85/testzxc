import os
import argparse
from openai import OpenAI
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", required=True, help="Batch ID to check")
    parser.add_argument("--api_key", default=None, help="OpenAI API Key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide --api_key or set OPENAI_API_KEY environment variable.")
        return

    client = OpenAI(api_key=api_key)
    batch_id = args.batch_id

    batch_obj = client.batches.retrieve(batch_id)
    print(f"Batch ID: {batch_id}")
    print(f"Status: {batch_obj.status}")
    print(f"Counts: {batch_obj.request_counts}")
    
    if batch_obj.status == "completed":
        print("\nBatch has completed! Downloading results...")
        output_file_id = batch_obj.output_file_id
        file_response = client.files.content(output_file_id)
        
        out_path = f"outputs/openai_batches/results_{batch_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(file_response.text)
        print(f"Results saved to {out_path}")
    elif batch_obj.status == "failed":
        print("\nBatch failed!")
        print(f"Errors: {batch_obj.errors}")

if __name__ == "__main__":
    main()
