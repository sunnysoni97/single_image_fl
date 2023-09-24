from strategy.common import cosine_annealing_round

if __name__ == "__main__":
    print("Testing cosine annealing")
    max_val = 500
    min_val = 50
    tot_rounds = 10

    for i in range(tot_rounds+1):
        print(f"Curr Round : {i} , Step Value : {int(cosine_annealing_round(max_lr=max_val, min_lr=min_val, max_rounds=tot_rounds, curr_round=i, restart_round=5, enable_restart=True))}")

    print("Test ended")
