
train:
	python train.py

pull-vec:
	wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
	- gunzip ./GoogleNews-vectors-negative300.bin.gz

clean:
	find . -name "*.pyc" -exec rm -f {} \;
