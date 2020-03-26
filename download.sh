wget -O demo_model_data/answers_lorra.txt https://dl.fbaipublicfiles.com/pythia/data/answers_textvqa_more_than_1.txt
wget -O demo_model_data/vocabulary_100k.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_100k.txt
wget -O demo_model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 
wget -O demo_model_data/lorra.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth
wget -O demo_model_data/lorra.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra.yml
wget -O demo_model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
wget -O demo_model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf demo_model_data/detectron_weights.tar.gz

pip install ninja yacs cython matplotlib demjson
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

rm -rf fastText
git clone https://github.com/facebookresearch/fastText.git fastText
cd ./fastText
pip install .
make

cd ..
rm -rf pythia
git clone https://github.com/facebookresearch/pythia.git pythia
cd pythia
pip install -e .

cd ..
mkdir content/pythia/pythia/.vector_cache
echo "Downloading fastText bin, this can take time."
wget -O /content/pythia/pythia/.vector_cache/wiki.en.bin https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin

git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build
python setup.py develop