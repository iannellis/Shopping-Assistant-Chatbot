#!/bin/bash

# Array of text values
texts=("show me shows with red color" 
"show me shows with blue color" 
"show me shows with green color")

# Base64-encoded image (empty in this case)
image=""

# Array to store elapsed times
elapsed_times=()

# Loop through the array and make a curl request for each text
for i in "${!texts[@]}"; do
  thread_id=$((i + 1))
  text="${texts[$i]}"
  
  start_time=$(date +%s%N)
  
  curl -X POST "http://3.86.148.142:9001/api/v1/prompt" \
       -H "Content-Type: application/json" \
       -d "{\"thread_id\": \"$thread_id\", \"text\": \"$text\", \"image\": \"$image\"}"
  
  end_time=$(date +%s%N)
  elapsed=$(( (end_time - start_time) / 1000000 ))
  
  elapsed_times+=($elapsed)
  echo "Request $i took $elapsed ms"
done

# Calculate mean elapsed time
total=0
for time in "${elapsed_times[@]}"; do
  total=$((total + time))
done
mean=$((total / ${#elapsed_times[@]}))

# Calculate 95th percentile
sorted_times=($(printf '%s\n' "${elapsed_times[@]}" | sort -n))
index=$(( (${#sorted_times[@]} * 95 + 99) / 100 - 1 ))
percentile_95=${sorted_times[$index]}

echo "Mean elapsed time: $mean ms"
echo "95th percentile elapsed time: $percentile_95 ms"
