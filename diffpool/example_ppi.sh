# AIDS - Diffpool
python -m train --bmname=AIDS --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=2 --num-classes=2 --method=soft-assign >> result_AIDS.txt
python -m train --bmname=AIDS --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=2 --num-classes=2 --method=soft-assign >> result_AIDS.txt


# DD - Diffpool
# python -m train --bmname=DD --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=2 --num-classes=2 --method=soft-assign >> result_DD.txt

# ENZYMES - Diffpool
# python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=2 --num-classes=6 --method=soft-assign >> result_ENZYMES.txt

# MUTAG - Diffpool
# python -m train --bmname=MUTAG --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=2 --num-classes=2 --method=soft-assign >> result_MUTAG.txt

# Mutagenicity - Diffpool
# python -m train --bmname=Mutagenicity --assign-ratio=0.1 --hidden-dim=32 --output-dim=32 --cuda=2 --num-classes=2 --method=soft-assign >> result_Mutagenicity.txt

# PTC_FM - Diffpool
python -m train_triplet --bmname=PTC_FM --assign-ratio=0.1 --hidden-dim=48 --output-dim=48 --cuda=3 --num-classes=2 --method=soft-assign --l2_regularize=0.01 --alpha=2.5 --epochs=10

# PTC_FM - Diffpool with regularization
python -m train_triplet --bmname=PTC_FM --assign-ratio=0.1 --hidden-dim=64 --output-dim=128 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0 --l2_regularize=0 --w=0 --gamma=0 --alpha=2 --epochs=10 >> results_PTC_FM.txt

python -m train_triplet --bmname=Mutagenicity --assign-ratio=0.1 --hidden-dim=64 --output-dim=128 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0 --l2_regularize=0 --w=0 --gamma=0 --alpha=2 --epochs=10 >> results_Mutagenicity.txt

name order: hidden dim - outdim - l1 - l2 - covariance weight - covariance decay gamma


python -m train_triplet --bmname=PTC_FM --assign-ratio=0.1 --hidden-dim=64 --output-dim=128 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0.1 --l2_regularize=0 --w=0 --gamma=0 --alpha=2 --epochs=3 >> results_PTC_FM.txt


python -m train_triplet --bmname=Mutagenicity --assign-ratio=0.1 --hidden-dim=64 --output-dim=128 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0 --l2_regularize=0 --w=0 --gamma=0 --alpha=2 --epochs=3 >> results_Mutagenicity.txt
python -m train_triplet --bmname=Mutagenicity --assign-ratio=0.1 --hidden-dim=64 --output-dim=128 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0.2 --l2_regularize=0 --w=0 --gamma=0 --alpha=2 --epochs=2 >> results_Mutagenicity.txt

python -m train_triplet --bmname=AIDS --assign-ratio=0.1 --hidden-dim=64 --output-dim=128 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0.2 --l2_regularize=0 --w=0 --gamma=0 --alpha=2 --epochs=2 >> results_AIDS.txt

python -m train_triplet --bmname=AIDS --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0.1 --l2_regularize=0 --w=0.2 --gamma=1 --alpha=2 --epochs=2 >> results_AIDS.txt

python -m train_triplet --bmname=AIDS --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0.1 --l2_regularize=0 --w=0.1 --gamma=1 --alpha=1 --epochs=2 >> results_AIDS.txt


python -m train_triplet --bmname=YeastH --assign-ratio=0.1 --hidden-dim=32 --output-dim=64 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0.1 --l2_regularize=0 --w=0.1 --gamma=1 --alpha=2 --epochs=1 >> results_YeastH.txt

python -m train_triplet --bmname=YeastH --assign-ratio=0.1 --hidden-dim=32 --output-dim=64 --cuda=3 --num-classes=2 --method=soft-assign --l1_regularize=0 --l2_regularize=0 --w=0.1 --gamma=1 --alpha=2 --epochs=1 >> results_YeastH.txt
