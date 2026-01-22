python sweep_fid.py \
    --results_folder /path/to/results \
    --data_folder /path/to/ground truth dataset path/ \
    --milestones 10000 50000 100000 150000 200000 250000 300000 350000 400000 450000 500000 550000 600000 \
    --csv_dir /path/to/fid saving directory \
    --csv_name fid file name, e.g., paso_fid.csv \
    --batch_size 512 \
    --num_samples 50000