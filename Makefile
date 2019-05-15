
init:
	- mkdir data
	- mkdir checkpoints
	pip install -r requirements.txt

train:
	cd src && python train.py

data:
	cd src && python preprocess.py

clean:
	find . -name "*.pyc" -exec rm -f {} \;
