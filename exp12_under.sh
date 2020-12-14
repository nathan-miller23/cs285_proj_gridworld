for DATASET_SIZE in 100 250 500 1000 2000 4000
do 
	python train.py -i tab_param -c -e 500 -empi -n $DATASET_SIZE -exp "delta_0.3_Tabular{$DATASET_SIZE}_underparam" --conv_arch 4 --fc_arch 100
	python train.py -i tab_param -c -e 500 -empi -n $DATASET_SIZE -adv -exp "delta_0.3_TabularAdv{$DATASET_SIZE}_underparam" --conv_arch 4 --fc_arch 100
done