path_image='./Houses-dataset/Houses Dataset'
path_struct_data='./Houses-dataset/Houses Dataset/HousesInfo.txt'
path_uszip_data='./data/uszips/uszips.csv'

method=CNN_FCNN #ViT_XGboost #CNN_FCNN

AUGM=--augm
DEBUG=--debug



setup:
	pip install -r requirements.txt

	wget https://simplemaps.com/static/data/us-zips/1.86/basic/simplemaps_uszips_basicv1.86.zip -O /tmp/uszips.zip
	unzip /tmp/uszips.zip -d ./data/uszips
	rm /tmp/uszips.zip

run: 
	python main.py $(path_image) $(path_struct_data) $(path_uszip_data) --method $(method)