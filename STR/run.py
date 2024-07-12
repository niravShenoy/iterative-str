import subprocess

subprocess.call('python STR/main.py --config ~/CISPA-projects/continuous_dst-2024/sparse_to_sparse/STR/configs/largescale/resnet18-cifar-iter-str-3.yaml --seed 3 --name cifar_0.3_ER_balanced_STR_Iter_01 --multigpu 0', shell=True)


# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-5.yaml --seed 3 --er-sparse-init 0.2 --name cifar_0.2_ER_str_induced_STR_3 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-1.yaml --seed 3 --er-sparse-init 0.2 --name cifar_0.2_ER_ERK_STR_3 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-2.yaml --seed 3 --er-sparse-init 0.2 --name cifar_0.2_ER_uniform_STR_3 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-3.yaml --seed 3 --er-sparse-init 0.2 --name cifar_0.2_ER_balanced_STR_3 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-4.yaml --seed 3 --er-sparse-init 0.2 --name cifar_0.2_ER_pyramidal_STR_3 --multigpu 0', shell=True)

# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-5.yaml --seed 2 --er-sparse-init 0.2 --name cifar_0.2_ER_str_induced_STR_2 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-1.yaml --seed 2 --er-sparse-init 0.2 --name cifar_0.2_ER_ERK_STR_2 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-2.yaml --seed 2 --er-sparse-init 0.2 --name cifar_0.2_ER_uniform_STR_2 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-3.yaml --seed 2 --er-sparse-init 0.2 --name cifar_0.2_ER_balanced_STR_2 --multigpu 0', shell=True)
# subprocess.call('python main.py --config configs/largescale/resnet18-cifar-str-4.yaml --seed 2 --er-sparse-init 0.2 --name cifar_0.2_ER_pyramidal_STR_2 --multigpu 0', shell=True)
