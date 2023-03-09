#!/bin/bash

# This script runs a set of iterations to execute the Octave script uniaxialCompression.m for different input values of Lx, Ly, Lz, and p.
# The results are stored in a CSV file named results_YYYY-MM-DD_HH-MM-SS.csv.

# Navigate to the folder ~/Documents/mamaML
cd ~/Documents/mamaML
#export ONSAS_PATH="/home/leopoldo/Documents/ONSAS.m/src"

# Create a timestamp to be used as the file name
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
filename="results_$timestamp.csv"

# Remove the previous output files if they exist
rm "$filename" > /dev/null 2>&1
rm cliOutput.txt > /dev/null 2>&1
rm output.txt > /dev/null 2>&1

# Define the Lx, Ly, Lz, and p values to be used in the iterations
Lx_values=(1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0)
Ly_values=(1)
Lz_values=(1)
E_values=(1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.4 3.5 3.6 3.7 3.8 3.9 4.0)
nu_values=(.3)
p_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0)

# Calculate the total number of iterations to be run
total_its=$(( ${#Lx_values[@]} * ${#Ly_values[@]} * ${#Lz_values[@]} * ${#p_values[@]} * ${#E_values[@]} * ${#nu_values[@]} )) 
current_its=0

# Print the conditions for this script
echo "Conditions for this script:"
echo "Lx values: ${Lx_values[@]}"
echo "Ly values: ${Ly_values[@]}"
echo "Lz values: ${Lz_values[@]}"
echo "E values: ${E_values[@]}"
echo "nu values: ${nu_values[@]}"
echo "p values: ${p_values[@]}"
echo "Total iterations: $total_its"

# Loop through all the Lx, Ly, Lz, and p values
for Lx in "${Lx_values[@]}"; do
  for Ly in "${Ly_values[@]}"; do
    for Lz in "${Lz_values[@]}"; do
      for E in "${E_values[@]}"; do
        for nu in "${nu_values[@]}"; do
          for p in "${p_values[@]}"; do
          # Run the Octave script with the specified input values
          LC_ALL=C octave -q ./../FEM_model/uniaixial_compression.m $Lx $Ly $Lz $E $nu $p > cliOutput.txt
          
          # Get the output values of Ux, Uy, and Uz
          Ux=$(sed -n 1p output.txt)
          Uy=$(sed -n 2p output.txt)
          Uz=$(sed -n 3p output.txt)
          
          # Increase the current iteration count
          current_its=$((current_its+1))
          
          # Print the current iteration count and the input values every 10 iterations
          if [ $((current_its % 10)) -eq 0 ]; then
            echo "Iteration $current_its of $total_its, Lx=$Lx, Ly=$Ly, Lz=$Lz, E=$E, nu=$nu, p=$p, Ux=$Ux, Uy=$Uy, Uz=$Uz"
          fi
          echo "$Lx,$Ly,$Lz,$E,$nu,$p,$Ux,$Uy,$Uz" >> "$filename"
          done
        done
      done
    done
  done
done

# Remove the temporary files
rm cliOutput.txt > /dev/null 2>&1
rm output.txt > /dev/null 2>&1
