#!/bin/bash

# This script runs a set of iterations to execute the Octave script uniaxialCompression.m for different input values of Lx, Ly, Lz, and p.
# The results are stored in a CSV file named results_YYYY-MM-DD_HH-MM-SS.csv.

# Navigate to the folder ~/Documents/mamaML
mkdir ./data/

# Create a timestamp to be used as the file name
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
filename="results_$timestamp.csv"

# Remove the previous output files if they exist
rm "$filename" > /dev/null 2>&1
rm cliOutput.txt > /dev/null 2>&1
rm output.txt > /dev/null 2>&1

# Define the Lx, Ly, Lz, and p values to be used in the iterations
# Geometry
Lx_values=(1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2)
Ly_values=(1)
Lz_values=(.5)
# Materials 
E1_values=(1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4)
# E1_values=(1.4)
nu1_values=(.3)
# E2_values=(1.4)
E2_values=(1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4)
nu2_values=(.3)
# Difne a linspace vector
p_values=(0.05 0.06 0.07 0.08 0.08 0.1 0.11 0.12 0.13 0.14)
# p_values=(0.05)


# Calculate the total number of iterations to be run
total_its=$(( ${#Lx_values[@]} * ${#Ly_values[@]} * ${#Lz_values[@]} * ${#p_values[@]} * ${#E1_values[@]} * ${#nu1_values[@]} * ${#E2_values[@]} * ${#nu2_values[@]} )) 
current_its=0

# Print the conditions for this script
echo "Conditions for this script:"
echo "Lx values: ${Lx_values[@]}"
echo "Ly values: ${Ly_values[@]}"
echo "Lz values: ${Lz_values[@]}"
echo "E1 values: ${E1_values[@]}"
echo "nu1 values: ${nu1_values[@]}"
echo "E2 values: ${E2_values[@]}"
echo "nu2 values: ${nu2_values[@]}"
echo "p values: ${p_values[@]}"
echo "Total iterations: $total_its"

# Loop through all the Lx, Ly, Lz, and p values
for Lx in "${Lx_values[@]}"; do
  for Ly in "${Ly_values[@]}"; do
    for Lz in "${Lz_values[@]}"; do
      for E2 in "${E2_values[@]}"; do
        for E1 in "${E1_values[@]}"; do
          for nu2 in "${nu2_values[@]}"; do
            for nu1 in "${nu1_values[@]}"; do
              for p in "${p_values[@]}"; do
                
                # Run the Octave script with the specified input values
                LC_ALL=C octave -q ./../FEM_model/cantilever_solid.m $Lx $E1 $E2 $p $nu1 $nu2 $Ly $Lz > cliOutput.txt
                
                # Get the output values of Ux, Uy, and Uz
                Ux=$(sed -n 1p output.txt)
                Uy=$(sed -n 2p output.txt)
                Uz=$(sed -n 3p output.txt)
                
                # Increase the current iteration count
                current_its=$((current_its+1))
              
                # Print the current iteration count and the input values every 10 iterations
                if [ $((current_its % 10)) -eq 0 ]; then
                  echo "Iteration $current_its of $total_its, Lx=$Lx, Ly=$Ly, Lz=$Lz, E1=$E1, nu1=$nu1, E1=$E1, nu1=$nu1 p=$p, Ux=$Ux, Uy=$Uy, Uz=$Uz"
                fi
                echo "$Lx, $Ly, $Lz, $E1, $nu1, $E2, $nu2, $p, $Ux, $Uy, $Uz" >> "$filename"
              done
            done
          done
        done
      done
    done
  done
done

# Remove the temporary files
rm cliOutput.txt > /dev/null 2>&1
rm output.txt > /dev/null 2>&1
