for learning_rate in 0.002 0.001 0.0001
do      
    python main.py --baseline=True --learning_rate=$learning_rate --hidden_size=100 --hidden_size_out=100 
done