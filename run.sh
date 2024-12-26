eval "$(conda shell.bash hook)"
conda activate genie_redux

# Initialize variables for named arguments
num_processes=${num_processes:-1}
config=${config:-"tokenizer.yaml"}

# Function to parse named arguments and kwargs
declare -A kwargs
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --num_processes=*) # Handle specific named argument
                num_processes="${1#*=}"
                ;;
            --config=*) # Handle specific named argument
                config="${1#*=}"
                ;;
            --*=*) # Handle dynamic key-value pairs
                key="${1%%=*}"
                key="${key#--}"
                value="${1#*=}"
                kwargs["$key"]="$value"
                ;;
            *)
                echo "Invalid argument: $1"
                exit 1
                ;;
        esac
        shift
    done

    # Display named arguments
    echo "Num Processes: $num_processes"
    echo "Config: $config"

    # Print parsed key-value pairs (kwargs)
    for key in "${!kwargs[@]}"; do
        echo "Key: $key, Value: ${kwargs[$key]}"
    done
}

# Call the function with all script arguments
parse_arguments "$@"

# Path to the main python script
python_script="main.py"


# prepare the arguments for the python script
args="+config=$config"
for key in "${!kwargs[@]}"; do
    args="$args ++config.$key=${kwargs[$key]}"
done

echo "Arguments: $args"

accelerate launch --num_processes=$num_processes --num_machines=1 --mixed_precision=bf16 $python_script $args