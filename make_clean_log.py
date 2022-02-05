
new_file = open('./mlp_log_file_epoch=4_lr=0.03_m=0.9_clean.log', 'w')
with open(
        "final_res/lr=0.03_opt=adam_batchSize=4_m=0.9_t=0.07/mlp_final/log_file_mlp_adam_lr=0.03_m=0.9_t=0.07_final.log", 'r') as f:
    for line in f:
        if 'end' in line or 'validation' in line:
            new_file.write(f'{line}')
        if 'mean' in line:
            new_file.write(f'\t{line}')
            if 'total' in line:
                new_file.write('\n')

