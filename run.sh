

# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type sgd
python cv_standard.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adam
python cv_standard.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adam
python cv_standard.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 64 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adam
python cv_standard.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 32 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adam

# w
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 30 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 30 --optimizer_type adam

# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 40 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 40 --optimizer_type adam

# python cv.py --dataset cifar10 --model_name vit --max_steps 5000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-4 --ema_decay 0.995 --training_mode serial --P 12 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name vit --max_steps 5000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-4 --ema_decay 0.995 --training_mode serial --P 12 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name vit --max_steps 5000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-4 --ema_decay 0.995 --training_mode serial --P 12 --optimizer_type adam


# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode serial --P 20 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode serial --P 20 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode serial --P 20 --optimizer_type adam

# python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adam


python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "LLM-Research/Llama-3.2-1B" --max_steps 1000 --batch_size 32 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 2
python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 30 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 4
python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 30 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 7
python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 32 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 10
python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 32 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 14
python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 32 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 17
python LLM_Task_CSV_Logger.py --local_model_dir "./model/LLM-Research/Llama-3.2-1B" --modelscope_name "Llama-3.2-1B" --max_steps 1000 --batch_size 32 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 19


python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 7
python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 14
python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 21
python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 28
python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 35
python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 42
python LLM_Task_CSV_Logger.py --local_model_dir "./model/gpt2" --modelscope_name "gpt2" --max_steps 1000 --batch_size 50 --learning_rate 6e-5 --threshold 1e-5 --ema_decay 0.9 --training_mode parallel --P 49


[Validation at step 500] Test Acc: 64.09% | Best Acc: 0.00%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 1.3371 | Acc: 52.22% | Time: 0:00:19:   5%|█████████▎                                                                                                                                                                             | 500/9800 [00:20<04:39, 33.27it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 1000] Test Acc: 70.92% | Best Acc: 64.09%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 1.1649 | Acc: 58.68% | Time: 0:00:32:  10%|██████████████████▌                                                                                                                                                                   | 1000/9800 [00:33<04:06, 35.71it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 1500] Test Acc: 72.67% | Best Acc: 70.92%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 1.0660 | Acc: 62.35% | Time: 0:00:44:  15%|███████████████████████████▊                                                                                                                                                          | 1500/9800 [00:45<03:28, 39.73it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 2000] Test Acc: 74.67% | Best Acc: 72.67%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.9982 | Acc: 64.80% | Time: 0:00:57:  20%|█████████████████████████████████████▏                                                                                                                                                | 2000/9800 [00:58<02:59, 43.39it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 2500] Test Acc: 75.27% | Best Acc: 74.67%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.9455 | Acc: 66.71% | Time: 0:01:09:  26%|██████████████████████████████████████████████▍                                                                                                                                       | 2500/9800 [01:10<02:46, 43.93it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 3000] Test Acc: 76.92% | Best Acc: 75.27%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.9037 | Acc: 68.22% | Time: 0:01:22:  31%|███████████████████████████████████████████████████████▋                                                                                                                              | 3000/9800 [01:23<02:47, 40.67it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 3500] Test Acc: 77.92% | Best Acc: 76.92%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.8683 | Acc: 69.49% | Time: 0:01:35:  36%|█████████████████████████████████████████████████████████████████                                                                                                                     | 3500/9800 [01:36<02:08, 48.97it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 4000] Test Acc: 78.17% | Best Acc: 77.92%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.8384 | Acc: 70.56% | Time: 0:01:48:  41%|██████████████████████████████████████████████████████████████████████████▎                                                                                                           | 4000/9800 [01:48<02:37, 36.83it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 4500] Test Acc: 78.39% | Best Acc: 78.17%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.8125 | Acc: 71.47% | Time: 0:02:00:  46%|███████████████████████████████████████████████████████████████████████████████████▌                                                                                                  | 4500/9800 [02:01<01:52, 47.01it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 5000] Test Acc: 78.65% | Best Acc: 78.39%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.7897 | Acc: 72.29% | Time: 0:02:12:  51%|████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                         | 5000/9800 [02:13<01:42, 46.61it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 5500] Test Acc: 79.19% | Best Acc: 78.65%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.7692 | Acc: 73.01% | Time: 0:02:25:  56%|██████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                               | 5500/9800 [02:26<01:58, 36.28it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 6000] Test Acc: 79.11% | Best Acc: 79.19%
Loss: 0.7508 | Acc: 73.66% | Time: 0:02:37:  61%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                      | 6000/9800 [02:38<01:41, 37.47it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 6500] Test Acc: 79.64% | Best Acc: 79.19%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.7342 | Acc: 74.25% | Time: 0:02:49:  66%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                             | 6500/9800 [02:50<01:14, 44.15it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 7000] Test Acc: 79.70% | Best Acc: 79.64%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.7190 | Acc: 74.78% | Time: 0:03:02:  71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                    | 7000/9800 [03:03<01:07, 41.22it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 7500] Test Acc: 79.88% | Best Acc: 79.70%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.7055 | Acc: 75.25% | Time: 0:03:14:  77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                          | 7500/9800 [03:15<00:52, 44.13it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 8000] Test Acc: 80.10% | Best Acc: 79.88%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.6928 | Acc: 75.70% | Time: 0:03:27:  82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                 | 8000/9800 [03:28<00:39, 45.97it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 8500] Test Acc: 80.32% | Best Acc: 80.10%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.6814 | Acc: 76.11% | Time: 0:03:40:  87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                        | 8500/9800 [03:41<00:26, 48.43it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 9000] Test Acc: 80.35% | Best Acc: 80.32%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.6710 | Acc: 76.47% | Time: 0:03:52:  92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏              | 9000/9800 [03:53<00:18, 44.10it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 9500] Test Acc: 80.35% | Best Acc: 80.35%
Loss: 0.6616 | Acc: 76.80% | Time: 0:04:05:  97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍     | 9500/9800 [04:05<00:07, 41.91it/s]/home/lu_24/PASO/cv_standard.py:859: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 9800] Test Acc: 80.42% | Best Acc: 80.35%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 0.6564 | Acc: 76.99% | Time: 0:04:12: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9800/9800 [04:13<00:00, 38.63it/s]

Training completed in 0:04:13
Final test accuracy: 80.42%

[Validation at step 1] Test Acc: 12.57% | Best Acc: 0.00%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
Loss: 2.3094 | Acc: 8.40% | Time: 0:00:03:   0%|                                                                                                                                                                                        | 1/9800 [00:28<8:36:33,  3.16s/it]/home/lu_24/PASO/cv_standard.py:544: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/home/lu_24/PASO/cv_standard.py:544: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/home/lu_24/PASO/cv_standard.py:544: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
Loss: 2.1549 | Acc: 22.52% | Time: 0:00:34:   0%|▊                                                                                                                                                                                       | 45/9800 [00:34<29:29,  5.51it/s]/home/lu_24/PASO/cv_standard.py:544: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
Loss: 2.0563 | Acc: 26.93% | Time: 0:00:41:   1%|█▌                                                                                                                                                                                      | 84/9800 [00:41<18:18,  8.85it/s]/home/lu_24/PASO/cv_standard.py:544: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
Loss: 1.9220 | Acc: 31.93% | Time: 0:00:51:   2%|███▏                                                                                                                                                                                   | 169/9800 [00:51<11:37, 13.80it/s]/home/lu_24/PASO/cv_standard.py:544: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
                                                                                                                                                                                                                                                                           
[Validation at step 501] Test Acc: 52.62% | Best Acc: 12.57%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 1004] Test Acc: 61.36% | Best Acc: 52.62%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 1506] Test Acc: 66.68% | Best Acc: 61.36%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 2007] Test Acc: 69.56% | Best Acc: 66.68%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 2512] Test Acc: 71.00% | Best Acc: 69.56%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 3012] Test Acc: 72.20% | Best Acc: 71.00%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 3514] Test Acc: 73.95% | Best Acc: 72.20%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 4021] Test Acc: 74.17% | Best Acc: 73.95%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 4525] Test Acc: 75.69% | Best Acc: 74.17%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 5029] Test Acc: 75.90% | Best Acc: 75.69%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 5533] Test Acc: 76.64% | Best Acc: 75.90%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 6037] Test Acc: 76.68% | Best Acc: 76.64%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 6541] Test Acc: 76.67% | Best Acc: 76.68%
                                                                                                                                                                                                                                                                           
[Validation at step 7045] Test Acc: 76.71% | Best Acc: 76.68%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 7549] Test Acc: 76.70% | Best Acc: 76.71%
                                                                                                                                                                                                                                                                           
[Validation at step 8053] Test Acc: 76.77% | Best Acc: 76.71%
Saving new best model to ./checkpoint                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                           
[Validation at step 8557] Test Acc: 76.68% | Best Acc: 76.77%
                                                                                                                                                                                                                                                                           
[Validation at step 9061] Test Acc: 76.52% | Best Acc: 76.77%
                                                                                                                                                                                                                                                                           
[Validation at step 9565] Test Acc: 76.55% | Best Acc: 76.77%
Loss: 0.7845 | Acc: 72.35% | Time: 0:05:08: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9800/9800 [05:08<00:00, 31.78it/s]
[W1116 23:49:23.592832863 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1116 23:49:23.612063667 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1116 23:49:24.905657714 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

Training completed in 0:05:08
Total communication time: 0:01:28
Final test accuracy: 76.42%
Final test precision: 76.85%
Final test recall: 76.42%
Final test f1-score: 76.46%
Total iterations: 1543 (vs 9800 normal iterations)
Effective speed-up: 6.35x                                                                                                                                                                                                                                
                                                            


[Validation at step 6000] Test Acc: 67.45% | Best Acc: 66.73%
Saving new best model to ./checkpoint                                                               
Loss: 1.1912 | Acc: 58.16% | Time: 0:03:03:  61%|█████████▏     | 6000/9800 [03:04<01:38, 38.43it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 6500] Test Acc: 67.50% | Best Acc: 67.45%
Saving new best model to ./checkpoint                                                               
Loss: 1.1766 | Acc: 58.71% | Time: 0:03:17:  66%|█████████▉     | 6500/9800 [03:18<01:16, 43.37it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 7000] Test Acc: 67.99% | Best Acc: 67.50%
Saving new best model to ./checkpoint                                                               
Loss: 1.1634 | Acc: 59.20% | Time: 0:03:32:  71%|██████████▋    | 7000/9800 [03:33<01:07, 41.42it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 7500] Test Acc: 67.61% | Best Acc: 67.99%
Loss: 1.1517 | Acc: 59.64% | Time: 0:03:47:  77%|███████████▍   | 7500/9800 [03:48<00:57, 40.05it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 8000] Test Acc: 67.97% | Best Acc: 67.99%
Loss: 1.1411 | Acc: 60.03% | Time: 0:04:02:  82%|████████████▏  | 8000/9800 [04:03<00:42, 42.02it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 8500] Test Acc: 68.08% | Best Acc: 67.99%
Saving new best model to ./checkpoint                                                               
Loss: 1.1316 | Acc: 60.40% | Time: 0:04:17:  87%|█████████████  | 8500/9800 [04:17<00:28, 45.46it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 9000] Test Acc: 68.05% | Best Acc: 68.08%
Loss: 1.1231 | Acc: 60.72% | Time: 0:04:31:  92%|█████████████▊ | 9000/9800 [04:32<00:18, 43.20it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 9500] Test Acc: 67.99% | Best Acc: 68.08%
Loss: 1.1153 | Acc: 61.01% | Time: 0:04:45:  97%|██████████████▌| 9500/9800 [04:46<00:08, 36.84it/s]/home/lu_24/PASO/cv_standard.py:861: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=config.use_amp):
                                                                                                    
[Validation at step 9800] Test Acc: 67.99% | Best Acc: 68.08%
Loss: 1.1111 | Acc: 61.17% | Time: 0:04:55: 100%|███████████████| 9800/9800 [04:56<00:00, 33.08it/s]

Training completed in 0:04:56
Final test accuracy: 67.99%

# Training parameters
batch_size = 512
n_epochs = 100     # Number of epochs (overrides max_steps if set)
max_steps = 1000    # Total iteration steps (used if n_epochs is None)
learning_rate = 0.0001
momentum = 0.9
optimizer_type = 'adam'
training_mode = 'parallel'

# Acceleration configuration (for parallel mode)
P = 6   # Window size
threshold = 0.0000001 # Error threshold 0.993
ema_decay = 0.993 # Threshold exponential moving average decay rate
adaptivity_type = 'mean'  # Adaptive strategy: 'mean' or 'median'
Training completed in 0:04:08
Total communication time: 0:01:39
Final test accuracy: 66.88%
Final test precision: 66.88%
Final test recall: 66.88%
Final test f1-score: 66.28%
Total iterations: 1824 (vs 9800 normal iterations)

[Validation at step 12063] Test Acc: 79.92% | Best Acc: 80.28%
                                                                                                                                                                                                 
[Validation at step 12567] Test Acc: 80.10% | Best Acc: 80.28%
                                                                                                                                                                                                 
[Validation at step 13070] Test Acc: 80.07% | Best Acc: 80.28%
                                                                                                                                                                                                 
[Validation at step 13571] Test Acc: 80.06% | Best Acc: 80.28%
                                                                                                                                                                                                 
[Validation at step 14075] Test Acc: 80.09% | Best Acc: 80.28%
                                                                                                                                                                                                 
[Validation at step 14576] Test Acc: 80.29% | Best Acc: 80.28%
Saving new best model to ./checkpoint                                                                                                                                                            
Loss: 0.6467 | Acc: 77.13% | Time: 0:03:37: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 14700/14700 [03:37<00:00, 67.50it/s]
[W1117 20:33:50.340774309 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1117 20:33:50.414422059 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1117 20:33:50.435025265 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1117 20:33:50.476054713 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
[W1117 20:33:50.537127724 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]

Training completed in 0:03:37
Total communication time: 0:01:57
Final test accuracy: 80.09%
Final test precision: 80.05%
Final test recall: 80.09%
Final test f1-score: 80.02%
Total iterations: 2881 (vs 14700 normal iterations)
Effective speed-up: 5.10x