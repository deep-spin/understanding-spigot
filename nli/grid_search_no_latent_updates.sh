for latent_model_type in marginals ste_marginals sparsemap argmax perturb_and_map
do      
    for learning_rate in 0.01 0.001 0.0001 0.00001
    do   
        python main.py --learning_rate=$learning_rate --hidden_size=200 --dropout=0.2 --latent_model_type=$latent_model_type 
    done
done