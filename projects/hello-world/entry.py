
import sklearn
import pickle
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
SOURCE_PKL = DATA_DIR / 'model.pkl'

classes = []
model = None
vectorizer = None

def load_pickle_data():
    global classes
    global model
    global vectorizer
    with open(SOURCE_PKL, 'rb') as f:
        loaded_pickle_obj = pickle.loads(f.read())
        classes = loaded_pickle_obj['classes']
        model = loaded_pickle_obj['model']
        vectorizer = loaded_pickle_obj['vectorizer']
        
        

load_pickle_data()

def label_predictor(text="Hello World"):
    global classes
    global model
    global vectorizer
    x_test = vectorizer.transform([text])
    target = model.predict(x_test) # target is an array of numpy.int64, we need a python `int` instead
    preds = {}
    for i, val in enumerate(target[0]):
        key_label = classes[i]
        preds[key_label] = int(val) # convert numpy.int64 into a python `int`
    return preds

def run(json_data={}, *args, **kwargs):
    '''
    Required method for tight.ai serving
    Returns a dictionary that is `json.dumps` ready.
    '''
    
    if 'question' not in json_data:
        return {'message': "a question is required", 'status': 400}
    input_question = json_data.get('question')
    tags = label_predictor(text=input_question)
    return {
        "question": input_question,
        "tags": tags
    }
    
