import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

class Backend:
    def __init__(self):
        # Load processed data
        self.train_df = pd.read_csv('./data/processed/train.csv')
        self.test_df = pd.read_csv('./data/processed/test.csv')

        # Fit vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, sublinear_tf=True)
        self.X_train = self.vectorizer.fit_transform(self.train_df['symptoms'])

        # Fit model
        self.model = LogisticRegression(max_iter=400, multi_class='ovr')
        self.model.fit(self.X_train, self.train_df['id'])

        # Mapping from id to diagnosis and treatment
        self.id_to_diagnosis = dict(zip(self.train_df['id'], self.train_df['Diagnosis']))
        self.id_to_treatment = dict(zip(self.train_df['id'], self.train_df['Treatment_Plan']))

    def predict(self, symptoms_text):
        '''
        Predict diagnosis and treatment from a list of symptoms
        '''
        X_input = self.vectorizer.transform([symptoms_text])
        pred_id = self.model.predict(X_input)[0]
        diagnosis = self.id_to_diagnosis.get(pred_id, 'Unknown')
        treatment = self.id_to_treatment.get(pred_id, 'Unknown')
        return diagnosis, treatment
