import streamlit as st


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
    'Cherbourg': 'C',
    'Queenstown': 'Q',
    'Southampton': 'S',
}


class App(object):
    def __init__(self):
        self.pclass = None
        self.sex = None
        self.title = None
        self.embarked = None
        self.age = None
        self.sibsp = None
        self.parch = None

    def render_page(self):
        st.title('Would you have survived the Titanic?')
        st.markdown(
            '''
            Based on a few details about yourself and the type of ticket that
            you would have bought to get on the ship, this will predict whether
            or not you would have survived the sinking of the Titanic in 1912.
            '''
        )

        self.pclass = st.radio('Ticket Class',
                          options=TICKET_CLASSES.keys(),
                          horizontal=True)
        self.sex = st.radio('Sex',
                       options=SEXES.keys(),
                       horizontal=True)
        self.title = st.radio('Your Title',
                         options=TITLES.keys(),
                         horizontal=True)
        self.embarked = st.radio('Port of Embarkation',
                            options=EMBARKED.keys(),
                            horizontal=True)

        self.age = st.slider('Age',
                        min_value=0,
                        max_value=80)
        self.sibsp = st.slider('Number of siblings or spouses aboard with you',
                          min_value=0,
                          max_value=10)
        self.parch = st.slider('Number of parents or children aboard with you',
                          min_value=0,
                          max_value=10)


if __name__ == '__main__':
    app = App()
    app.render_page()
