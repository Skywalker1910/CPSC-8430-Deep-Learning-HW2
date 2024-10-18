# hw2_seq2seq.ps1
# PowerShell version of hw2_seq2seq.sh

# Parameters
$data_dir = $args[0]
$output_file = $args[1]

# Run the Python inference script
python run_inference.py --data_dir $data_dir --output $output_file

