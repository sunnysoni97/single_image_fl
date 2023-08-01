from strategy.common import cosine_annealing_round

if __name__ == "__main__":
    max_rounds = 10
    max = 0.5
    min = 0.1
    for i in range(0,max_rounds+1):
        no = cosine_annealing_round(max_lr=max, min_lr=min,max_rounds=(max_rounds-1),curr_round=(i-1))
        print(f'{i} = {no}')
