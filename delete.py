# import joblib
# import os
# import pprint

# from globalVars import WAV2VEC2_MODEL_NAME


# def make_safe_folder_name(name):
#     """
#     Convert a name into a safe folder name.

#     Examples:
#         "jonatasgrosman/wav2vec2-large-xlsr-53-greek"
#         -> "jonatasgrosman__wav2vec2-large-xlsr-53-greek"

#         "word with spaces" -> "word_with_spaces"
#     """

#     safe = name.replace("/", "__")
#     safe = safe.replace(" ", "_")
#     return safe



# model_name = WAV2VEC2_MODEL_NAME
# embedding_layer = 12
# word = "δεσμεύω"


# safe_model_name = make_safe_folder_name(model_name)
# safe_word = make_safe_folder_name(word)

# model_output_folder = os.path.join(
#     "outputs",
#     safe_model_name,
#     f"layer_{embedding_layer}",
#     safe_word,
# )
# report_folder = os.path.join(model_output_folder, "report")
# model_path = os.path.join(report_folder, "..", "model_oneclass_" + safe_word + ".joblib")
# data = joblib.load(model_path)

# # If it's a Pandas DataFrame, save it as a text-readable CSV string
# # (or just print the first 20 rows to paste into Claude)
# print("--- CLAUDE HANDOFF DATA ---")
# print(f"Dictionary Keys: {list(data.keys())}\n")

# for key, value in data.items():
#     print(f"Key: {key} | Type: {type(value)}")
    
#     # If it's a trained model, get its configurations
#     if hasattr(value, "get_params"):
#         pprint.pprint(value.get_params())
        
#     # IF IT'S A NUMPY ARRAY / MATRIX: Print its shape instead of raw numbers
#     elif hasattr(value, "shape"):
#         print(f"  -> Array Shape: {value.shape} | Data Type: {value.dtype}")
        
#     # If it's a list, print its length
#     elif isinstance(value, list):
#         print(f"  -> List Length: {len(value)} items")
#         print(f"  -> First item preview: {value[0] if value else 'Empty'}")
        
#     else:
#         print(f"  -> Value Preview: {str(value)[:300]}")
#     print("-" * 50)


import winsound

print("Playing the file 'Welcome.wav'")

# winsound.PlaySound('filename', flag)
winsound.PlaySound('Welcome.wav', winsound.SND_FILENAME)