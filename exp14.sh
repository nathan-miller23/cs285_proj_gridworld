python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.3 -o out_seed
for SEED in 1111 2222 3333 4444 5555
do 
	python train.py -i out_seed -c -e 300 -s $SEED -n 250 -empi -exp "delta_0.3_Tabular_N_250_SEED_$SEED"
	python train.py -i out_seed -c -e 300 -s $SEED -n 250 -empi -adv -exp "delta_0.3_TabularAdv_N_250_SEED_$SEED"
done