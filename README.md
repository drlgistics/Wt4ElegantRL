# Wt4ElegantRL
Wt4ElegantRL = WonderTrader(https://github.com/wondertrader/wtpy) + ElegantRL(https://github.com/AI4Finance-Foundation/ElegantRL)

## INSTALL
clone
```
git clone https://github.com/drlgistics/Wt4ElegantRL.git
cd Wt4ElegantRL

conda create -n wt4elegantrl python==3.9.7 #python>3.7,<3.10
conda activate wt4elegantrl
```

with cuda
```
conda install -n wt4elegantrl -c conda-forge -c pytorch --file l --file ./requirements/minimize_with_cuda.txt
```

without cuda
```
conda install -n wt4elegantrl -c conda-forge -c pytorch --file l --file ./requirements/minimize_without_cuda.txt
```