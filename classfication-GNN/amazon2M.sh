python -u multiclass.py --dataset Amazon2M --agp_alg appnp_agp  --alpha 0.2 --rmax 1e-8 --L 20 --lr 0.01 --dropout 0.1 --hidden 1024 --batch 100000
python -u multiclass.py --dataset Amazon2M --agp_alg gdc_agp  --ti 4 --rmax 1e-8 --L 20 --lr 0.01 --dropout 0.1 --hidden 1024 --batch 100000
python -u multiclass.py --dataset Amazon2M --agp_alg sgc_agp --rmax 1e-10 --L 10 --lr 0.01 --dropout 0.1 --hidden 1024 --batch 100000
python -u multiclass.py --dataset Amazon2M --agp_alg appnp_agp  --alpha 0.2 --rmax 1e-10 --L 4 --lr 0.01 --dropout 0.1 --hidden 1024 --batch 100000