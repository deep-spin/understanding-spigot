for latent_model_type in marginals ste_marginals sparsemap argmax perturb_and_map
do      
    for learning_rate in 0.00005
    do
        python main.py --learning_rate=$learning_rate --hidden_size=100 --hidden_size_out=100 --latent_grad_updates=1 --latent_step_size=1 --latent_model_type=$latent_model_type
    done
done