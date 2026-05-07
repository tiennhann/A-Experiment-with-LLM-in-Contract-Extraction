import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "..", "..", "contract-nli")
DATA_DIR = os.path.join(BASE_DIR, "data")

def remove_keys_from_documents(input_file, output_file):
    # List of keys to remove
    keys_to_remove = ['id','spans','annotation_sets', 'document_type', 'url']
    try:
        # Load the JSON data from the input
        with open(input_file, 'r', encoding='utf8') as infile:
            json_data = json.load(infile)
        
        # Iterate through each document and remove the specified keys
        for document in json_data.get('documents', []):
            for key in keys_to_remove:
                if key in document:
                    del document[key]
        if 'labels' in json_data:
            del json_data['labels']

        # Write modified JSON data to the output file
        with open(output_file, 'w', encoding='utf8') as outfile:
            json.dump(json_data, outfile, indent=4, ensure_ascii=False)
        
        print(f"Modiffied JSON data has been written to {output_file}.")

    except Exception as e:
        print(f"An error occured: {e}")


input_file_test  = os.path.join(RAW_DIR,  "test.json")
output_file_test = os.path.join(DATA_DIR, "test_modified.json")

input_file_train  = os.path.join(RAW_DIR,  "train.json")
output_file_train = os.path.join(DATA_DIR, "train_modified.json")

remove_keys_from_documents(input_file_test, output_file_test)
remove_keys_from_documents(input_file_train, output_file_train)



