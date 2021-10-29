# Wt4ElegantRL
Wt4ElegantRL = [WonderTrader](https://github.com/wondertrader/wtpy) + [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)

## INSTALL
clone
```
git clone https://github.com/drlgistics/Wt4ElegantRL.git
cd Wt4ElegantRL
```

with cuda
```
conda create -n wt4elegantrl -f ./requirements/full_with_cuda.yaml
```

without cuda
```
conda create -n wt4elegantrl -f ./requirements/full_without_cuda.yaml
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

## ELEGANTRL
train
```
~~python ./compare_elegantrl.py train~~
```