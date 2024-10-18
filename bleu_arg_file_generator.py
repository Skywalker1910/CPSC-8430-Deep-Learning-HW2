# Load video_ids and generated captions
with open('MLDS_hw2_1_data/testing_data/id.txt', 'r') as f_vid:
    video_ids = [line.strip() for line in f_vid.readlines()]

with open('output.txt', 'r') as f_out:
    generated_captions = [line.strip() for line in f_out.readlines()]

# Ensure the lengths of both files are the same
if len(video_ids) != len(generated_captions):
    raise ValueError("The number of video IDs and generated captions do not match.")

# Combine and write to a new output file with video_id,captions format
with open('captions.txt', 'w') as f_combined:
    for video_id, caption in zip(video_ids, generated_captions):
        f_combined.write(f"{video_id},{caption}\n")

print("Combined output saved to captions.txt")
