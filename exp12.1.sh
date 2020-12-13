python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.3 -o out_tab_big
for DATASET_SIZE in 2000 4000
do 
	python train.py -i out_tab_big -c -e 300 -empi -n $DATASET_SIZE -exp "delta_0.3_Tabular{$DATASET_SIZE}"
	python train.py -i out_tab_big -c -e 300 -empi -n $DATASET_SIZE -adv -exp "delta_0.3_TabularAdv{$DATASET_SIZE}"
done
