for learning_rate in 0.001 0.0001 0.00001
do      
    python main.py --baseline=True --learning_rate=$learning_rate --hidden_size=200 --dropout=0.2
done