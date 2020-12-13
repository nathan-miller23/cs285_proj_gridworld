python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.3 -o out_rbg -rbg
for DATASET_SIZE in 100 250 500 1000 2000 4000 8000
do
	for SEED in 1111 2222 
	do
		python train.py -i out_rbg -c -e 500 -empi -n $DATASET_SIZE -exp "delta_0.3_Tabular{$DATASET_SIZE}_rbg_overparam_Seed_{$SEED}" -rbg --conv_arch 8 16 32 64 128 --fc_arch 100 200 400
		python train.py -i out_rbg -c -e 500 -empi -n $DATASET_SIZE -adv -exp "delta_0.3_TabularAdv{$DATASET_SIZE}_rbg_overparam_Seed_{$SEED}" -rbg --conv_arch 8 16 32 64 --fc_arch 100 200 400
	done
done

for DATASET_SIZE in 100 250 500 1000 2000 4000 8000
do
	for SEED in 1111 2222
	do
		python train.py -i out_rbg -c -e 500 -empi -n $DATASET_SIZE -exp "delta_0.3_Tabular{$DATASET_SIZE}_Seed_{$SEED}" -rbg
		python train.py -i out_rbg -c -e 500 -empi -n $DATASET_SIZE -adv -exp "delta_0.3_TabularAdv{$DATASET_SIZE}_Seed_{$SEED}" -rbg
	done
done

for DATASET_SIZE in 100 250 500 1000 2000 4000 8000
do 
	for SEED in 1111 2222
	do
		python train.py -i out_rbg -c -e 500 -empi -n $DATASET_SIZE -exp "delta_0.3_Tabular{$DATASET_SIZE}_Seed_{$SEED}_underparam" -rbg --conv_arch 4 --fc_arch 100
		python train.py -i out_rbg -c -e 500 -empi -n $DATASET_SIZE -adv -exp "delta_0.3_TabularAdv{$DATASET_SIZE}_Seed_{$SEED}_underparam" -rbg --conv_arch 4 --fc_arch 100
	done
done