import os
import argparse
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the chunked JSONL file")
    parser.add_argument("--api_key", default=None, help="OpenAI API Key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please provide --api_key or set OPENAI_API_KEY environment variable.")
        return

    client = OpenAI(api_key=api_key)
    file_path = args.file

    print(f"1. Uploading file: {file_path}")
    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(
          file=f,
          purpose="batch"
        )
    file_id = batch_input_file.id
    print(f"File uploaded successfully! File ID: {file_id}")

    print("2. Submitting Batch Request...")
    batch_obj = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": f"Evaluation batch for {os.path.basename(file_path)}"
        }
    )
    
    batch_id = batch_obj.id
    print(f"Batch submitted successfully! Batch ID: {batch_id}")
    print("\n--- NEXT STEPS ---")
    print(f"To check status, run: python src/eval/check_openai_batch.py --batch_id {batch_id} --api_key YOUR_KEY")
    print("Or view the status on the dashboard: https://platform.openai.com/batches")

if __name__ == "__main__":
    main()
