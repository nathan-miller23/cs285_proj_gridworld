for EPS in 0.0 0.001
do
    python generate_data.py --save_agent --save_environment -stdeps 0.4 -criteps $EPS --delta 0.3 -o "the_real_tightrope_$EPS" -n 5000 -tight
    python train.py -i "the_real_tightrope_$EPS" -c -e 300 -n 2500 -empi -exp "the_real_epoch_{300}_tightrop_eps_{$EPS}_SEED_{1}_n_{2500}"
    for LAMB in 0.0 1.0 5.0 20.0 100.0
    do   
        python train.py -i "the_real_tightrope_$EPS" -c -e 300 -n 2500 -quad -lam $LAMB -empi -adv -deep_q -exp "the_real_epoch_{300}_tightrop_eps_{$EPS}_lambda_{$LAMB}_SEED_1_n_{2500}_ddqn"
    done
done