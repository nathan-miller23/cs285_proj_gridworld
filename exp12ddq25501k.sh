python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.7 -o out
for DATASET_SIZE in 250 500 1000
do 
	#python train.py -deep_q -i out -c -e 300 -n $DATASET_SIZE -s 1 -exp "Tabular{$DATASET_SIZE}"
	python train.py -deep_q -i out -c -e 300 -n $DATASET_SIZE -s 1 -adv -exp "TabularAdv{$DATASET_SIZE}s1"
done

python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.7 -o out
for DATASET_SIZE in 250 500 1000
do 
	#python train.py -deep_q -i out -c -e 300 -n $DATASET_SIZE -s 2 -exp "Tabular{$DATASET_SIZE}"
	python train.py -deep_q -i out -c -e 300 -n $DATASET_SIZE -s 2 -adv -exp "TabularAdv{$DATASET_SIZE}s2"
done

python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.7 -o out
for DATASET_SIZE in 250 500 1000
do 
	#python train.py -deep_q -i out -c -e 300 -n $DATASET_SIZE -s 3 -exp "Tabular{$DATASET_SIZE}"
	python train.py -deep_q -i out -c -e 300 -n $DATASET_SIZE -s 3 -adv -exp "TabularAdv{$DATASET_SIZE}s3"
done