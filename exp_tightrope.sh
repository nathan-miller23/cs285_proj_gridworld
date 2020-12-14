for EPS in 0.01 0.05 0.1 0.25
do
    python generate_data.py --save_agent --save_environment -stdeps 0.4 -criteps $EPS --delta 0.3 -o "new_tightrope_$EPS" -n 5000
    for LAMB in 0.0 1.0 5.0 10.0 20.0
    do   
        for SEED in 111
        do
            python train.py -i "new_tightrope_$EPS" -c -e 300 -n 1000 -empi -exp "epoch_{300}_tightrop_eps_{$EPS}_lambda_{$LAMB}_SEED_{$SEED}_n_{1000}"
            python train.py -i "new_tightrope_$EPS" -c -e 300 -n 1000 -quad -lam $LAMB -empi -adv -deep_q -exp "epoch_{300}_tightrop_eps_{$EPS}_lambda_{$LAMB}_SEED_{$SEED}_n_{1000}_ddqn"
        done
    done
done