python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.7 -o out
for DATASET_SIZE in 2000 4000
do 
	python train.py -i out -c -e 300 -n $DATASET_SIZE -exp "Tabular{$DATASET_SIZE}"
	python train.py -i out -c -e 300 -n $DATASET_SIZE -adv -exp "TabularAdv{$DATASET_SIZE}"
done