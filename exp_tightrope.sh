for EPS in 0.0 0.01 0.05 0.1 0.25
do
    python generate_data.py --save_agent --save_environment -stdeps 0.4 -criteps $EPS --delta 0.3 -o "tightrope_$EPS" -n 5000
    for LAMB in 0.0 1.0 5.0 10.0 20.0
    do   
        for SEED in 111 222 
        do
            python train.py -i "tightrope_$EPS" -c -e 500 -n 2500 -empi -exp "tightrop_eps_{$EPS}_lambda_{$LAMB}_SEED_{$SEED}_n_{2500}"
            python train.py -i "tightrope_$EPS" -c -e 500 -n 2500 -quad -lam $LAMB -empi -adv -deep_q -exp "tightrop_eps_{$EPS}_lambda_{$LAMB}_SEED_{$SEED}_n_{2500}_ddqn"
        done
    done
done