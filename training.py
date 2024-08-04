import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from multiprocessing import Process, Manager

def train_model(data, labels, result_dict):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    # Store the results in a shared dictionary
    result_dict['model'] = model
    result_dict['score'] = score * 100

if __name__ == "__main__":
    # Load data
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # Use a Manager to create a shared dictionary for storing results
    with Manager() as manager:
        result_dict = manager.dict()

        # Create a process for training the model
        training_process = Process(target=train_model, args=(data, labels, result_dict))
        training_process.start()
        training_process.join()

        # Access the results from the shared dictionary
        model = result_dict['model']
        score = result_dict['score']

        print('{}% of samples were classified correctly!'.format(score))

        # Save the model
        with open('model.p', 'wb') as f:
            pickle.dump({'model': model}, f)
