import tools
import evaluation
model = tools.load_model()
evaluation.evalrank(model, data='coco', split='test')
# import demo, tools, datasets
# net = demo.build_convnet()
# train = datasets.load_dataset('coco', load_train=True)[0]
# vectors = tools.encode_sentences(model, train[0], verbose=False)
# demo.retrieve_captions(model, net, train[0], vectors, 'test_img.jpg', k=5)