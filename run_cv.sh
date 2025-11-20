#batch sweep
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.999 --training_mode parallel --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 2048 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 256 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 64 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 512 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 256 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type adam
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 64 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type adam

python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.97 --training_mode parallel --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 2048 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 256 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 64 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode parallel --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 512 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 256 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 128 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type sgd
python cv_v2.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 64 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.997 --training_mode serial --P 7 --optimizer_type sgd



# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 30 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 30 --optimizer_type adam

# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 40 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 40 --optimizer_type adam

python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 2048 --learning_rate 0.001 --threshold 1e-4 --ema_decay 0.995 --training_mode serial --P 12 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name vit --max_steps 5000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-4 --ema_decay 0.995 --training_mode serial --P 12 --optimizer_type sgd
python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size  4096 --learning_rate 0.001 --threshold 1e-4 --ema_decay 0.995 --training_mode serial --P 12 --optimizer_type adam


# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode serial --P 20 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode serial --P 20 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name cnn --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode serial --P 20 --optimizer_type adam

# python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adamw
# python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type sgd
# python cv.py --dataset cifar10 --model_name vit --max_steps 60000 --batch_size 4096 --learning_rate 0.001 --threshold 1e-5 --ema_decay 0.995 --training_mode parallel --P 20 --optimizer_type adam

