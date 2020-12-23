for latent_model_type in marginals ste_marginals sparsemap argmax perturb_and_map
do      
    for learning_rate in 0.002 0.001 0.0005 0.0001 0.00005 0.00002 0.00001
    do
        python main.py --learning_rate=$learning_rate --hidden_size=100 --hidden_size_out=100 --latent_grad_updates=1 --latent_step_size=1 --latent_model_type=$latent_model_type
    done
done