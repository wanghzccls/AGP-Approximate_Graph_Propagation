python -u papers100M.py --dataset papers100M --agp_alg sgc_agp --L 10 --rmax 1e-7 --lr 0.0001 --dropout 0.3 --hidden 2048
python -u papers100M.py --dataset papers100M --agp_alg appnp_agp --alpha 0.2 --L 20 --rmax 1e-7 --lr 0.0001 --dropout 0.3 --hidden 2048
python -u papers100M.py --dataset papers100M --agp_alg gdc_agp --ti 4 --L 20 --rmax 1e-7 --lr 0.0001 --dropout 0.3 --hidden 2048
python -u papers100M.py --dataset papers100M --agp_alg gdc_agp --ti 4 --L 3 --rmax 1e-8 --lr 0.0001 --dropout 0.3 --hidden 2048