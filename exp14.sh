python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.7 -o out
for SEED in 1111 2222 3333 4444 5555
do 
	python train.py -i out -c -e 300 -shuff -s $SEED -n 250 -exp "Tabular_N_250_SEED_$SEED"
	python train.py -i out -c -e 300 -shuff -s $SEED -n 250 -adv -exp "TabularAdv_N_250_SEED_$SEED"
done