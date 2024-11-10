import os
import time
import anthropic
import pandas as pd
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path
import re
from dataclasses import dataclass
from llava.instruct.instruct_postprocess import format_conv, clean_conv


class ConversationExtractor:
    def extract_conversation(self, response: str, pair_id: str) -> List[Dict]:
        """
        Extract all user and assistant messages from the conversation.
        Returns a list of ConversationTurn objects.
        """
        try:
            formatted_conv = format_conv({
                'result': response,
                'pair_id': pair_id,
                'domain': {'cervical_Spine': True, 'fracture': True, 'ct_scan': True},
            })
            return clean_conv(formatted_conv)
        except Exception as e:
            print(f"cannot extra {response}: {e}")
            return []


class ClaudeJSONProcessor:
    def __init__(self, api_key: str, input_file: str, prompt_template: str,
                 input_column: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the processor with necessary parameters.

        Args:
            api_key (str): Anthropic API key
            input_file (str): Path to input CSV file
            prompt_template (str): Template for generating prompts. Use {value} as placeholder
            input_column (str): Name of the column to process
            model (str): Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.input_file = input_file
        self.prompt_template = prompt_template
        self.input_column = input_column
        self.model = model

    def generate_prompt(self, caption: str) -> str:
        """Generate a prompt using the template and value."""
        return self.prompt_template.format(caption=caption)

    def get_claude_response(self, prompt: str, max_retries: int = 3) -> Dict[str, str]:
        """
        Get response from Claude with retry mechanism and better error handling.
        Returns a dictionary with response details.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1}/{max_retries}")

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1.0,
                )

                response = message.content[0].text
                if not response:
                    raise ValueError("Empty response received from Claude")

                return {
                    "status": "success",
                    "response": response,
                    "attempts": attempt + 1
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "status": "error",
                        "response": f"Failed after {max_retries} attempts: {str(e)}",
                        "attempts": max_retries
                    }
                print(f"Error: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def process_data(self) -> str:
        """
        Process the CSV file row by row and save results incrementally in JSON.
        Returns the path to the output file.
        """
        # Read input CSV
        df = pd.read_csv(self.input_file)

        # Validate input column exists
        if self.input_column not in df.columns:
            raise ValueError(f"Column '{self.input_column}' not found in CSV file")

        # Generate output filename
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"claude_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = []

        # Process rows one by one
        total_rows = len(df)
        print(f"\nProcessing {total_rows} rows...")

        extractor = ConversationExtractor()
        round = 2

        for index, row in df.iterrows():
            try:
                input_text = str(row[self.input_column])
                if len(input_text) < 50:
                    continue
                prompt = self.generate_prompt(input_text)
                print(f"prompt: {prompt}")

                print(f"\nProcessing row {index + 1}/{total_rows}")
                print(f"Input: {input_text[:100]}...")  # Show truncated input

                # Get response from Claude
                response_data = self.get_claude_response(prompt)
                print(f"response: {response_data}")  # Show truncated input
                id = f"{row['pmcid']}_{row['figureid']}" 
                result = extractor.extract_conversation(
                    response_data['response'],
                    id
                )
                if not result:
                    print(f"formatting result is empty!")
                    continue

                # second round
                result = result[0]
                result['id'] = f"{id}_{round}"

                # Create result entry
                results.append(result)

                # Save after each successful response
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(f"Response saved for row {index + 1}")

                # Optional: Add delay to respect rate limits
                time.sleep(1)  # Adjust as needed

            except Exception as e:
                error_msg = f"Error processing row {index + 1}: {str(e)}"
                print(f"\nERROR: {error_msg}")

                # Add error entry
                results["results"].append({
                    "row_index": index,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "input": input_text,
                    "prompt": prompt,
                    "response": {
                        "status": "error",
                        "response": error_msg,
                        "attempts": 0
                    }
                })

                # Save after error
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                continue

        print(f"\nProcessing complete! Results saved to: {output_file}")
        return str(output_file)

def main():
    # Configuration
    API_KEY = os.environ['ANTROPIC_API_KEY']
    INPUT_FILE = 'cervical_spine_figures-3.csv'
    INPUT_COLUMN = 'caption'

    PROMPT_TEMPLATE = """
You are an AI assistant specialized in biomedical topics. Specifically, radiology, CT, orthopaedics and cervical spine.

You are provided with a text description (Figure Caption) of a figure image from a medical research paper. The images typically feature CT scans in various planes depicting cervical spine(s) / fragments of such as well diagrams illustrating the pathologies associated such as fractures. The description / caption will include the details on the condition as well as any other info such as diagnosis, notes and recommendations relevant to the particular illustration.

Your task is to generate a hypothetical conversation between a person (User) inquiring about the image and you (Assistant) responding to their questions. The conversation should proceed as though both the User and Assistant are viewing the image, while not referring to the text information (Figure Caption and Figure Context).

# Requirements:
Below are requirements for generating the questions and answers in the conversation:

 - Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information.
 - Where applicable use the terminology associated with radiology and orthopaedics
 - Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead, refer to the information as being "in the image."
 - Ensure that questions are diverse and cover a range of visual aspects of the image.
 - The conversation should include at least 2-3 turns of questions and answers about the visual aspects of the image.
 - Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information. Encourage the user to consult a healthcare professional for advice.
 - Follow below example, so not extra explanations.
 - Include key words from the caption.

# Example response:
User: Where is the lesion located?
Assistant: The lesion is located in the posterior aspect of the right upper lobe of the lung.
User: Is it extending into the chest wall?
Assistant: Yes, the CT scan suggests that the large cavitating lesion in the right upper lobe is likely extending into the chest wall.
User: Can you explain more about the cavitating lesion?
Assistant: Certainly. A cavitating lesion refers to an abnormal mass or growth within the lung tissue that has a cavity or hollow space inside it. This cavity can be formed due to the necrosis (death) of the tissue inside the lesion, leading to the breakdown and disintegration of the affected tissue. Cavitating lesions can be caused by various factors, such as infections (like tuberculosis or fungal infections), autoimmune diseases, or malignancies (cancer). In the context of the CT scan provided, the large cavitating lesion in the right upper lobe of the lung is suggestive of a malignant (cancerous) growth. The cavity within the lesion may be a result of the rapid and uncontrolled growth of cancer cells, which can lead to the destruction of surrounding healthy tissue. It's important to note that further "diagnostic tests, such as a biopsy, would be required to confirm the malignancy and determine the specific type of cancer involved.

# The caption:
{caption}
    """

    try:
        # Initialize processor
        processor = ClaudeJSONProcessor(
            api_key=API_KEY,
            input_file=INPUT_FILE,
            prompt_template=PROMPT_TEMPLATE,
            input_column=INPUT_COLUMN
        )

        print("\nProcessing CSV file...")
        output_file = processor.process_data()
        print(f"\nProcessing complete! Results saved to: {output_file}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()

