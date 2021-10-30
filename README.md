# Wt4ElegantRL
Wt4ElegantRL = [WonderTrader](https://github.com/wondertrader/wtpy) + [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)

## INSTALL
clone
```
git clone https://github.com/drlgistics/Wt4ElegantRL.git
cd Wt4ElegantRL
conda create -n wt4elegantrl
```

with cuda
```
conda env create -n wt4elegantrl --file ./requirements/full_with_cuda.yaml
```

without cuda
```
conda env create -n wt4elegantrl --file ./requirements/full_without_cuda.yaml
```

activate
```
conda activate wt4elegantrl
```

## RLLIB
demo
```
python ./compare_rllib.py test -p ./trained/rllib/TD3_2021-10-29_11-21-37/TD3_SimpleCTAEnv_536e7_00000_0_2021-10-29_11-21-37/checkpoint_000023/checkpoint-23
```

train
```
python ./compare_rllib.py train
```

## SB3
demo
```
python ./compare_sb3.py test -p ./trained/sb3/best_model
```

train
```
python ./compare_sb3.py train
```

## ELEGANTRL
~~train~~
~~python ./compare_elegantrl.py train~~