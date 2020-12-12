python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.7 -o out_rbg -rbg
for DATASET_SIZE in 250 500 1000 2000 4000 8000
do 
	python train.py -i out_rbg -c -n $DATASET_SIZE -exp "Tabular{$DATASET_SIZE}_rbg_overparam" -rbg --conv_arch 8 16 32 64 --fc_arch 100 200 
	python train.py -i out_rbg -c -n $DATASET_SIZE -adv -exp "TabularAdv{$DATASET_SIZE}_rbg_overparam" -rbg --conv_arch 8 16 32 64 --fc_arch 100 200 
done

for DATASET_SIZE in 250 500 1000 2000 4000 8000
do 
	python train.py -i out_rbg -c -n $DATASET_SIZE -exp "Tabular{$DATASET_SIZE}_rbg" -rbg
	python train.py -i out_rbg -c -n $DATASET_SIZE -adv -exp "TabularAdv{$DATASET_SIZE}_rbg" -rbg
done

for DATASET_SIZE in 250 500 1000 2000 4000 8000
do 
	python train.py -i out_rbg -c -n $DATASET_SIZE -exp "Tabular{$DATASET_SIZE}_rbg_logistic_reg" -rbg --conv_arch  --fc_arch 
	python train.py -i out_rbg -c -n $DATASET_SIZE -adv -exp "TabularAdv{$DATASET_SIZE}_rbg_logistic_reg" -rbg --conv_arch --fc_arch 
done