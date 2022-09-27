import numpy as np
import pandas as pd
import pickle
import streamlit as st
from collections import OrderedDict


TICKET_CLASSES = {
    'Upper': 1,
    'Middle': 2,
    'Lower': 3,
}
SEXES = {
    'Female': 0,
    'Male': 1,
}
TITLES = {
    'Master': 0,
    'Miss': 1,
    'Mr': 2,
    'Mrs': 3,
}
EMBARKED = {
    'Cherbourg': 0,
    'Queenstown': 1,
    'Southampton': 2,
}
AGE_BUCKETS = OrderedDict({
    5: 0,
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    100: 5,
})


class App(object):

    def __init__(self, model_dir='models/titanic_model.sav'):
        self.form_pclass = None
        self.form_sex = None
        self.form_title = None
        self.form_embarked = None
        self.form_age = None
        self.form_sibsp = None
        self.form_parch = None
        self.model = App.load_model(model_dir)

    def render_page(self):
        st.title('Would you have survived the Titanic?')
        st.markdown(
            '''
            Based on a few details about your self, this will predict whether
            or not you would have survived the sinking of the Titanic in 1912.
            '''
        )

        with st.form(key='profile_form'):
            self.form_pclass = st.radio('Ticket Class',
                              options=TICKET_CLASSES.keys(),
                              horizontal=True)
            self.form_sex = st.radio('Sex',
                           options=SEXES.keys(),
                           horizontal=True)
            self.form_title = st.radio('Your Title',
                             options=TITLES.keys(),
                             horizontal=True)
            self.form_embarked = st.radio('Port of Embarkation',
                                options=EMBARKED.keys(),
                                horizontal=True)
            self.form_age = st.slider('Age',
                            min_value=0,
                            max_value=100)
            self.form_sibsp = st.slider(
                            'Number of siblings or spouses aboard with you',
                            min_value=0,
                            max_value=10)
            self.form_parch = st.slider(
                            'Number of parents or children aboard with you',
                            min_value=0,
                            max_value=10)

            submitted = st.form_submit_button('Submit')
            if submitted:
                self.get_prediction()

    def get_prediction(self):
        pclass = TICKET_CLASSES.get(self.form_pclass)
        sex = SEXES.get(self.form_sex)
        title = TITLES.get(self.form_title)
        embarked = EMBARKED.get(self.form_embarked)
        age = self.form_age
        sibsp = self.form_sibsp
        parch = self.form_parch

        # Age needs to be bucketed
        age_bucket = 0
        for k,v in AGE_BUCKETS.items():
            if age <= k:
                age_bucket = v
                break

        data = (
            np
            .array([pclass, sex, age_bucket, sibsp, parch, embarked, title])
            .reshape(1,-1)
        )

        res = self.model.predict(data)[0]

        if res == 1:
            output_colour = 'green'
            output_text = 'Congrats! You survived!'
        else:
            output_colour = 'red'
            output_text = 'Sorry, you died.'

        st.write(f'<p style="color:{ output_colour };">{ output_text }</p>',
                 unsafe_allow_html=True)

    @staticmethod
    def load_model(model_dir):
        try:
            model = None
            with open(model_dir, 'rb') as file:
                model = pickle.load(file)
            return model
        except:
            st.error('Unable to load prediction model!')
            raise Exception


if __name__ == '__main__':
    app = App()
    app.render_page()
