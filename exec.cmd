@REM python train.py --env_name k4free ^
@REM                 --exp_name k4free_n17 ^
@REM                 --N 17 ^
@REM                 --encoding_tokens single_integer ^
@REM                 --alpha_mode exact ^
@REM                 --approx_restarts 50 ^
@REM                 --max_len 250 ^
@REM                 --temperature 0.8 ^
@REM                 --inc_temp 0.05 ^
@REM                 --gensize 50000 ^
@REM                 --num_samples_from_model 200000 ^
@REM                 --pop_size 100000 ^
@REM                 --batch_size 32 ^
@REM                 --n_layer 4 ^
@REM                 --n_head 8 ^
@REM                 --n_embd 256 ^
@REM                 --gen_batch_size 400 ^
@REM                 --max_steps 12000


python train.py --env_name k4free ^
                --exp_name k4free_n29 ^
                --exp_id 2026_04_02_17_59_04 ^
                --N 29 ^
                --encoding_tokens single_integer ^
                --alpha_mode approx ^
                --approx_restarts 200 ^
                --max_len 250 ^
                --temperature 0.6 ^
                --inc_temp 0.1 ^
                --gensize 100000 ^
                --num_samples_from_model 200000 ^
                --pop_size 200000 ^
                --batch_size 32 ^
                --n_layer 4 ^
                --n_head 8 ^
                --n_embd 256 ^
                --gen_batch_size 300 ^
                --max_steps 15000
                @REM --process_pool false